import asyncio
import logging
import sys
import os
import json
import re

from .config import settings, get_local_model, get_gpt4_1_mini
from .search_system import AdvancedSearchSystem
from .utilties.search_utilities import invoke_with_timeout_and_retry

# 配置基础日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_local_model_health(timeout: float = 2.0) -> bool:
    """检测本地模型(vLLM)是否可用"""
    print("🔍 Checking local model connectivity...", end=" ", flush=True)
    try:
        llm = get_local_model(temperature=0.1)
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = timeout
        llm.invoke("Hi") 
        print("✅ ONLINE")
        return True
    except Exception as e:
        print(f"❌ OFFLINE (Error: {str(e)[:50]}...)")
        return False

def read_context_from_file(file_path: str) -> str:
    """Helper: 从文件读取治疗方案内容"""
    try:
        file_path = file_path.strip('"').strip("'")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return ""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading context file: {e}")
        return ""

def parse_graph_ec_report(raw_text: str):
    """
    分离 PathoRAG 生成的正文和参考文献，并用正则安全提取最大文献序号
    """
    separator = "==================== 参考文献 (References) ===================="
    max_index = 0
    ref_text = ""
    report_body = raw_text
    
    if separator in raw_text:
        parts = raw_text.split(separator)
        report_body = parts[0].strip()
        ref_text = parts[1].strip()
        
        # 匹配 "[数字] PMID:" 或 "[数字] DocID:" 或 "[数字] URL:"
        matches = re.findall(r'\[(\d+)\]\s*(?:PMID|DocID|URL)', ref_text)
        if matches:
            max_index = max([int(m) for m in matches])
            
    return report_body, max_index, ref_text, separator

async def extract_structured_task(raw_text: str, fast_llm) -> dict:
    """
    无损结构化翻译官：负责将 PathoRAG 传来的【初步治疗方案 + 患者全息数据】完美拆解。
    新增核心逻辑：对并发症进行“预后级”与“次要偶发级”的分流。
    """
    print(" [Parser] Extracting patient profile, proposed plan, and stratifying comorbidities...")
    prompt = f"""
你是一个极其严格且具备顶尖临床思维的【医疗数据结构化专家】。
你的任务是从 PathoRAG 传出的前置报告（包含患者原始特征和初始 MDT 方案）中，无损提取所有关键信息。

=========================================
【👇需要你提取的病历与初始方案数据👇】
{raw_text}
=========================================

🚨 【核心分类与提取红线】：
1. **肿瘤核心数据（oncology_core）**：绝对无损地提取分期、病理、分子分型。
2. **【极度重要】并发症分级（Comorbidity Stratification）**：
   - **重大合并症 (major_comorbidities)**：必须提取对【肿瘤预后、化疗耐受性、靶向药物毒理】有重大影响的疾病（如：糖尿病、肥胖、高血压、冠心病/心血管疾病、肾衰竭、肝硬化、自身免疫病等）。
   - **次要异常 (incidental_findings)**：提取局灶性、轻度且暂不影响全身肿瘤系统治疗的异常（如：浅表胃炎、轻度脂肪肝、肺部散在慢性炎症、主动脉壁钙化等）。
3. **PICO深度问题**：准确提取原报告末尾留给 PathoEBM 深度查证的问题。

【强制输出的 JSON 嵌套模板】（仅输出 JSON，不要有其他废话）：
{{
  "oncology_core": {{
    "basic_info": "年龄、绝经状态、体能评分等",
    "diagnosis_and_stage": "完整的术后诊断及FIGO分期",
    "pathology_and_molecular": "病理类型、浸润深度、淋巴结、LVSI、MMR、p53等标志物"
  }},
  "comorbidities": {{
    "major_comorbidities": [
      "重大合并症1 (例如: 20年糖尿病史)",
      "重大合并症2 (例如: 冠心病支架植入后)"
    ],
    "incidental_findings": [
      "次要异常1 (例如: 两肺底散在慢性炎症)",
      "次要异常2 (例如: 胃窦糜烂HP感染)"
    ]
  }},
  "proposed_plan": {{
    "main_oncology_treatment": "初步给出的肿瘤核心放化疗方案细节",
    "follow_up_schedule": "随访计划"
  }},
  "clinical_questions_for_ebm": [
    "提取报告末尾要求查证的具体PICO临床问题1",
    "问题2"
  ]
}}
"""
    try:
        response = await invoke_with_timeout_and_retry(fast_llm, prompt, timeout=1200.0)
        raw_resp = response.content
        
        # 剔除可能存在的思维链并精准截取 JSON
        cleaned_resp = re.sub(r"<think>.*?</think>", "", raw_resp, flags=re.DOTALL | re.IGNORECASE).strip()
        print(cleaned_resp)
        start_idx = cleaned_resp.find('{')
        end_idx = cleaned_resp.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            cleaned_resp = cleaned_resp[start_idx:end_idx+1]
        
        structured_data = json.loads(cleaned_resp, strict=False)
        print("✅ [Parser] Structured data successfully extracted and stratified.")
        return structured_data
    except Exception as e:
        print(f"⚠️ [Parser] Parsing failed: {e}. Fallback to default structure.")
        return {
            "oncology_core": {"raw": "Extraction failed, rely on raw text."},
            "comorbidities": {"major_comorbidities": [], "incidental_findings": []},
            "proposed_plan": {"main": "Validate the proposed treatment plan"},
            "clinical_questions_for_ebm": ["Check latest evidence for the given plan"]
        }

async def run_evidence_update(treatment_context: str):
    """
    执行核心循证更新。
    路由策略：重大合并症送入Deep Search检索毒性与预后，次要异常直接生成转诊话术。
    """
    if check_local_model_health():
        print("🚀 Using Local vLLM Model (Free & Private).")
        current_mode = "local"
        fast_llm = get_local_model(temperature=0.1)
    else:
        print("⚠️ Local model unavailable. Switching to Cloud API (DeepSeek/GPT).")
        current_mode = "deepseek"
        try: 
            fast_llm = get_gpt4_1_mini()
        except: 
            fast_llm = get_local_model(temperature=0.1)

    # 1. 拆分图谱初步报告的主体与参考文献
    report_body, max_index, baseline_refs, separator = parse_graph_ec_report(treatment_context)
    print(f"✅ [Parser] Found {max_index} baseline references from graph-ec.")

    # 2. LLM 结构化拆解与合并症分级
    structured_task = await extract_structured_task(treatment_context, fast_llm)
    structured_task["baseline_references"] = {"max_index": max_index}
    
    # 🚨 核心深搜 Payload 组装：
    # 将 oncology_core 和 major_comorbidities 组装在一起交给 Deep Search
    # 丢弃 incidental_findings 以防干扰主检索链路
    search_payload = {
        "oncology_profile": structured_task.get("oncology_core", {}),
        "major_comorbidities_affecting_treatment": structured_task.get("comorbidities", {}).get("major_comorbidities", []),
        "preliminary_plan": structured_task.get("proposed_plan", {}),
        "specific_pico_questions": structured_task.get("clinical_questions_for_ebm", []),
        "baseline_references": {"max_index": max_index}
    }
    
    print(f"\n🔄 Clinical Evidence Update System Activated.")
    print(f"   Context Length: {len(report_body)} characters")
    print(f"   Major Comorbidities to Analyze: {len(search_payload['major_comorbidities_affecting_treatment'])}")
    print("   Targeting Sources: PubMed (2024+), ClinicalTrials.gov\n")

    # 严格限制加载的工具列表，防止 Token 爆炸导致超时
    my_target_tools = [
        "search_recent_pubmed",  
        "get_studies",           
        "get_adverse_reactions_by_drug_name", 
        "get_warnings_and_cautions_by_drug_name" 
    ]

    # 3. 初始化深搜系统
    system = AdvancedSearchSystem(
        max_iterations=settings.detailed.iteration, 
        questions_per_iteration=settings.detailed.questions_per_iteration,
        is_report=True,
        treatment_context=report_body, 
        structured_task=search_payload, # 👈 只喂给它肿瘤和重大合并症数据
        using_model=current_mode,
        chosen_tools=my_target_tools
    )

    try:
        await system.initialize()
        
        query = "Please validate the preliminary treatment plan, carefully assess the impact of major comorbidities (if any) on drug toxicity and overall survival, and answer the specific clinical questions provided."
        results = await system.analyze_topic(query)
        print(f"\n✅ Evidence synthesis task completed.")
        
        if results.get("final_report"):
             print("\n" + "="*60)
             print("   FINAL COMBINED CLINICAL REPORT   ")
             print("="*60 + "\n")
             
             final_resp_text = results['final_report']
             
             # 4. 完美拼接与【次要异常的拦截转诊】
             new_evidence_text = final_resp_text
             new_refs_text = ""
             split_marker = "=================================================="
             
             if split_marker in final_resp_text:
                 parts = final_resp_text.split(split_marker)
                 new_evidence_text = parts[0].strip()
                 new_refs_text = parts[1].strip()
             
             # 获取在前端被截留的次要异常，自动生成轻量级分诊模块
             incidental_findings = structured_task.get("comorbidities", {}).get("incidental_findings", [])
             referral_section = ""
             if incidental_findings:
                 referral_section = "\n### 其他非肿瘤异常及随访建议\n"
                 for idx, item in enumerate(incidental_findings, 1):
                     referral_section += f"{idx}. **关于[{item}]**：建议转诊至相应专科门诊进一步评估治疗方案。\n"
             
             # 将重写后的终极正文、次要并发症转诊、旧文献列表、新文献列表按序无缝缝合
             combined_report = f"### 🏥 循证校验与优化的最终治疗方案 (Deep EBM Synthesized Plan)\n\n" \
                               f"{new_evidence_text}\n" \
                               f"{referral_section}\n" \
                               f"{separator}\n" \
                               f"{baseline_refs}\n"
             
             if new_refs_text:
                 combined_report += f"{new_refs_text}\n"
             
             print(combined_report)
             
             # 保存到文件
             report_path = "evidence_update_report.md"
             with open(report_path, "w", encoding="utf-8") as f:
                 f.write(combined_report)
             print(f"\n📄 Report saved to: {os.path.abspath(report_path)}")

    except Exception as e:
        logger.error(f"Run failed: {e}")
        print(f"\n❌ Error during execution: {e}")

async def main():
    """主程序入口"""
    print("==================================================")
    print("   OriGene Clinical Evidence Validator (Auto-Hybrid)")
    print("==================================================")
    print("Strategy: Extract Context -> Deep Search -> Auto Gap-Closing Citations")
    print("Type 'quit' to exit at any time.")

    while True:
        print("\n--------------------------------------------------")
        print("Select Input Method:")
        print("1) Paste Treatment Plan Text (Markdown with References)")
        print("2) Load Plan from File (.txt/.md)")
        
        choice = input("\nEnter number (1 or 2): ").strip()
        
        if choice.lower() == 'quit':
            break

        treatment_context = ""

        if choice == "2":
            path = input("Enter file path: ").strip()
            if path.lower() == 'quit': break
            treatment_context = read_context_from_file(path)
        elif choice == "1":
            print("\n👇 Please paste the Clinical Treatment Plan (Markdown) below.")
            print("Type 'END' on a new line when finished:\n")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            treatment_context = "\n".join(lines)
        else:
            print("Invalid selection. Please enter 1 or 2.")
            continue
        
        if not treatment_context.strip():
            print("❌ Empty context provided. Please try again.")
            continue

        # 执行核心任务
        await run_evidence_update(treatment_context)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
