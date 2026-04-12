import asyncio
import json
import logging
import os
import re
import textwrap
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

from .config import (
    settings,
    get_claude_openai,
    get_deepseek_r1,
    get_deepseek_v3,
    get_gpt4_1,
    get_gpt4_1_mini,
    get_local_model,
)
from .connect_mcp import OrigeneMCPToolClient, mcp_servers
from .search_system_support import (
    compress_all_llm,
    extract_and_convert_list,
    parse_single,
    safe_json_from_text,
    SourcesReference,
)
from .tool_executor import ToolExecutor
from .tool_selector import ToolSelector
from .utilties.search_utilities import (
    invoke_with_timeout_and_retry,
    write_log_process_safe,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(
    log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

GLOBAL_LLM_SEMAPHORE = None
GLOBAL_API_SEMAPHORE = None

def get_global_semaphores():
    global GLOBAL_LLM_SEMAPHORE, GLOBAL_API_SEMAPHORE
    if GLOBAL_LLM_SEMAPHORE is None:
        GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(8)  # 你的显卡极限
    if GLOBAL_API_SEMAPHORE is None:
        GLOBAL_API_SEMAPHORE = asyncio.Semaphore(10)  # PubMed 的限流红线
    return GLOBAL_LLM_SEMAPHORE, GLOBAL_API_SEMAPHORE


def remove_think_tags(text: str) -> str:
    """Robustly remove <think> tags from model output."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned:
        cleaned = re.sub(r".*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class ReferencePool:
    """Reference pool for citations, supporting baseline offset."""
    def __init__(self, baseline_max_index: int = 0) -> None:
        self.pool: List[SourcesReference] = []
        self.link2idx: dict[str, int] = {}
        self.base_idx = baseline_max_index # 记录 graph-ec 传来的最大文献序号

    def add(self, title: str, citation: str, link: str) -> int:
        if not link:
            return -1
        if link in self.link2idx:
            return self.link2idx[link]
        # 新文献从 base_idx + 1 开始接力编号
        idx = self.base_idx + len(self.pool) + 1
        self.link2idx[link] = idx
        self.pool.append(
            SourcesReference(title=title or link, subtitle=citation or "", link=link)
        )
        return idx
    
    def get_ref_by_idx(self, idx: int):
        # 换算回本地 pool 的实际索引
        actual_idx = idx - self.base_idx - 1
        if 0 <= actual_idx < len(self.pool):
            return self.pool[actual_idx]
        return None


class AdvancedSearchSystem:
    def __init__(
        self,
        max_iterations=2,
        questions_per_iteration=8,
        is_report=True,
        chosen_tools: list[str] = None,
        error_log_path: str = "",
        using_model = "deepseek",  
        treatment_context: str = "",
        structured_task: dict = None, # 接收从 main.py 传来的结构化任务
    ):
        self.structured_task = structured_task or {}
        
        # 初始化带偏移量的引用池
        baseline_refs = self.structured_task.get("baseline_references", {})
        max_idx = baseline_refs.get("max_index", 0)
        self.ref_pool = ReferencePool(baseline_max_index=max_idx) 
        
        self.chosen_tools = chosen_tools
        self.is_report = is_report
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration
        self.treatment_context = treatment_context
        self.knowledge_chunks = []
        self.all_links_of_system = []
        self.questions_by_iteration = {}

        if error_log_path == "":
            error_log_path = os.path.join(
                log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        self.error_log_path = error_log_path

        # === 模型初始化逻辑 ===
        self.using_model = using_model
        
        if self.using_model == "local":
            logger.info("🤖 Using Local vLLM Model (DeepSeek-R1-32B / qwen-test)")
            try:
                local_llm = get_local_model(temperature=0.1)
                local_fast_llm = get_local_model(temperature=0.1) 
                
                self.model = local_llm              
                self.reasoning_model = local_llm    
                self.tool_planning_model = local_llm 
                self.report_model = local_llm       
                self.fast_model = local_fast_llm    
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise e

        elif self.using_model == "deepseek":
            self.model = get_deepseek_r1()
            self.reasoning_model = get_deepseek_r1()
            self.tool_planning_model = get_deepseek_v3()
            self.fast_model = get_deepseek_v3()
            self.report_model = get_deepseek_r1()
            
        else:
            self.model = get_gpt4_1()
            self.reasoning_model = get_gpt4_1()
            self.tool_planning_model = get_gpt4_1()
            self.fast_model = get_gpt4_1_mini()
            self.report_model = get_gpt4_1()

    async def initialize(self):
        """Initialize tools in Pure API Mode (Lightweight)."""
        try:
            self.mcp_tool_client = OrigeneMCPToolClient(mcp_servers, self.chosen_tools)
            await self.mcp_tool_client.initialize()
            self.mcp_tool_dict = self.mcp_tool_client.tool2source
            
            self.tool_selector = ToolSelector(
                self.tool_planning_model,
                self.reasoning_model,
                self.mcp_tool_client,
                tool_info_data=None,
                embedding_api_key=None,
                embedding_cache=None,
                available_tools=self.chosen_tools,
            )
            
            self.tool_executor = ToolExecutor(
                self.mcp_tool_client, self.error_log_path, self.fast_model
            )
            
            logger.info("✅ System initialized in PURE API MODE (Official Databases Only).")
            
        except Exception as e:
            logger.error(f"Failed to initialize search system: {e}")
            raise e

    async def _get_follow_up_questions(self, current_knowledge: str, query: str) -> List[str]:
        now = datetime.now().strftime("%Y-%m-%d")
        structured_data = json.dumps(self.structured_task, ensure_ascii=False, indent=2)
        
        prompt = f"""
        你是一名顶级的“循证医学检索转化专家”（Clinical Evidence Coordinator）。
        上游系统已经为你提取好了【患者的核心结构化数据】，包含初步治疗方案、合并症以及PICO问题。

        【🚨 核心检索策略：三足鼎立（极度重要！）】：
        你的并发检索额度上限为 {self.questions_per_iteration} 个检索词。你必须充分利用这些额度，**绝对不能只查 PICO 问题！**你的检索词必须全面覆盖以下三个维度：

        1. **【灯塔试验复核（最高优先级，占用 1-3 个额度）】**：
           - 根据患者的具体分期，严格从以下灯塔库中挑选对应的重磅试验进行检索。
           - 早期高危及局部晚期（III、IVA期）：必须同时核查 PORTEC-3 和 GOG-0258。
           - 晚期（IVB期）及复发一线：必须核查 GOG-209, NRG-GY018, RUBY。
           - 晚期复发（二线及以上）：必须核查 KEYNOTE-775。
           - ⚠️ 检索词示例：PORTEC-3 AND ("survival"[Title/Abstract] OR "outcomes"[Title/Abstract]) AND 2018:2026[dp]

        2. **【主干方案深度审阅（核心纠偏，占用 2-3 个额度）】（🚨 此前被你忽略的重点！）**：
           - 你必须直接审视【初步治疗方案（main_oncology_treatment）】中提出的具体放疗参数（如外照射、阴道近距离放疗）、具体化疗方案（如 TC、TAC 方案等）。
           - 针对该方案在当前患者分期和病理（如浆液性癌、G3）下的疗效、安全性以及是否为标准治疗（Standard of Care），生成直接的验证性检索词！
           - ⚠️ 检索词示例：("endometrial cancer" OR "endometrial carcinoma") AND "serous" AND ("Stage III" OR "Stage IIIA") AND "TC regimen" AND "radiotherapy" AND 2018:2026[dp]

        3. **【合并症质询与高价值PICO（占用 1-2 个额度）】**：
           - 提取结构化数据中遗留的 PICO 问题，剔除无意义的边缘问题（如缺乏定论的标记物）。
           - 针对患者真实的重大合并症（如冠心病、肾功能不全），生成关于药物毒性替换（如 卡铂 AND 顺铂 AND 头对头对比）或剂量调整的检索词。

        【上游传入的结构化数据】：
        {structured_data}
        
        【当前已验证的知识】：
        {current_knowledge}

        【强制输出 JSON 格式】：
        你必须输出合法的 JSON 格式。
        ```json
        {{
            "discovery_reasoning": "简述你如何分配这 {self.questions_per_iteration} 个检索额度，以确保全面覆盖灯塔试验、主干方案审阅和合并症质询。",
            "sub_queries": [
                "灯塔试验检索词1",
                "灯塔试验检索词2 (若有)",
                "主干方案具体放化疗参数审阅词1",
                "主干方案具体放化疗参数审阅词2",
                "合并症/毒性替代/PICO检索词1"
            ]
        }}
        ```
        💡 请直接输出 JSON 代码块，确保充分利用 {self.questions_per_iteration} 个并发额度，不可遗漏对主干方案的深度核查！
        """

        try:
            response = await invoke_with_timeout_and_retry(
                self.tool_planning_model, prompt, timeout=1200.0, max_retries=3
            )
            
            response_text = remove_think_tags(response.content)
            
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = safe_json_from_text(response_text[json_start:json_end])
                if parsed:
                    # 打印主动质疑的结果，方便后台监控
                    challenge = parsed.get("clinical_challenge", "")
                    logger.info(f"💡 [翻译官临床质询与高级语法控制]: {challenge}")
                    
                    questions = parsed.get("sub_queries", [])
                    return questions[:self.questions_per_iteration]
            
            return extract_and_convert_list(response_text)[:self.questions_per_iteration]
            
        except Exception as e:
            logger.warning(f"Failed to generate questions: {e}")
            return []
        
    async def _answer_query(
        self,
        current_knowledge: str,
        query: str,
        current_iteration: int,
        max_iterations: int,
    ) -> str:
        """Synthesize API findings."""
        existing_refs = [
            f"[{idx}] {ref.link} — {ref.title}"
            for idx, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1)
        ]
        refs_block = "\n".join(existing_refs) or "*None yet*"

        prompt = textwrap.dedent(f"""
        ## Task: Evidence Synthesis
        
        Validating clinical plan using **OFFICIAL API DATA** (PubMed/FDA/CT.gov).
        
        ## Verified API Data
        {current_knowledge}
        
        ## Sources
        {refs_block}
        
        ## Instructions
        1. **Validate**: Does 2024-Present evidence support the focus areas?
        2. **Update**: Identify newer trials/approvals.
        3. **Detail**: Note specific regimens (Drug/Dose) found in evidence.
        
        ## Output Template
        ## Evidence Status
        - **Decision Point**: [e.g. Adjuvant Therapy]
          - **Status**: [✅ Supported / ⚠️ Controversy / ❓ No Data]
          - **Key Evidence**: [Summarize findings from PubMed/CT.gov [^^n]]
        
        ## Missing Data
        [What couldn't be verified?]
        """)

        try:
            response = await invoke_with_timeout_and_retry(
                self.model, prompt, timeout=1200.0, max_retries=3
            )
            content = remove_think_tags(response.content)
            return content
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            return "Error synthesizing evidence."

    async def _run_prognosis_retrieval_track(self) -> str:
        """
        🚀 完全独立的预后检索流水线：
        专设检索词 -> 独立调用PubMed -> 放宽限制提取 Top 10 -> 生成纯净专属预后文献库
        """
        logger.info("🧬 [Prognosis Track] 启动独立预后数据检索分支...")
        
        # 1. 提取结构化肿瘤信息生成精准检索词
        oncology_core = self.structured_task.get("oncology_profile", {})
        if isinstance(oncology_core, dict):
            diag = oncology_core.get("diagnosis_and_stage", "")
            patho = oncology_core.get("pathology_and_molecular", "")
            base_info = f"{diag} {patho}"
        else:
            base_info = str(oncology_core)
            
        # 🚀 [修改点 1]：增加“分期泛化（降维）”逻辑，大幅提升召回率
        prompt = textwrap.dedent(f"""
        你是一名专业的 PubMed 检索词生成专家。
        请根据以下患者信息，生成一个【精简、高召回率且带有时效性】的预后生存率检索词（Boolean string）。
        
        【患者信息】：
        {base_info}
        
        【🚨 严苛规则（极度重要）】：
        1. **核心要素**：你的检索词只能包含：疾病大类名称 + 核心病理类型 + 分期。
        2. **【分期泛化降维】（最核心红线！）**：由于医学文献摘要极少精确到子分期，如果患者的分期带有细分字母或数字（如 IIIA1期、IIIC2期），你**必须**将其降维简化为大分期（如 "Stage III" OR "Stage IIIA"）写入检索词！绝对禁止在检索词中保留 "IIIA1" 这种极度具体的子分期，否则会导致查无文献！
        3. **拒绝过度拟合**：绝对不要把具体的突变（如 p53, PR-, Ki-67）或具体的转移部位塞进检索词里！
        4. **【强制年份限制与后缀】**：必须在末尾加上：AND ("survival"[Title/Abstract] OR "prognosis"[Title/Abstract]) AND 2018:2026[dp]
        
        【正确示范（注意分期的降维处理）】：
        ("endometrial cancer" OR "endometrial carcinoma") AND "serous" AND ("Stage III" OR "Stage IIIA") AND ("survival"[Title/Abstract] OR "prognosis"[Title/Abstract]) AND 2018:2026[dp]
        
        请直接输出最终的英文检索词字符串，不要有任何多余的解释，不要带引号。
        """)
        
        try:
            resp = await invoke_with_timeout_and_retry(self.fast_model, prompt, timeout=120.0)
            prog_query = remove_think_tags(resp.content).strip().replace('"', '')
        except Exception:
            prog_query = "Endometrial cancer survival rate"

        logger.info(f"🧬 [Prognosis Track] 专属预后检索词: {prog_query}")
        
        # 2. 独立调用检索工具
        local_selector = ToolSelector(
            self.tool_planning_model, self.reasoning_model, self.mcp_tool_client, available_tools=["search_recent_pubmed"]
        )
        local_executor = ToolExecutor(
            self.mcp_tool_client, self.error_log_path, self.fast_model
        )
        
        try:
            # 🚀 [修改点 1]：向工具选择器下达强硬命令，强制拉取大量文献！
            force_command = (
                f"CRITICAL: Search for survival rates using this EXACT query: {prog_query}. "
                f"You MUST set the retrieval parameter (e.g. max_results, top_k, or retmax) to at least 5 "
                f"to ensure we have a large enough cohort data pool."
            )
            t_calls = await local_selector.run(force_command)
            t_results = await local_executor.run(t_calls) or []
        except Exception as e:
            logger.error(f"Prognosis search failed: {e}")
            return "检索预后文献失败。"
            
        # 3. 结果解析与文献库挂载 (🚨 放宽限制到 10 篇)
        prog_evidence = ""
        count = 0
        
        for res in t_results:
            res_str = str(res).replace('\\n', '\n')
            blocks = res_str.split("\n---\n") if "\n---\n" in res_str else [res_str]
            for block in blocks:
                if not block.strip() or "Unknown Title" in block: continue
                
                url = ""
                pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', block, re.IGNORECASE)
                if pmid_match: url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
                
                title_match = re.search(r'Title:\s*([^\n]+)', block, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "Prognosis Study"
                
                if url and url not in self.all_links_of_system:
                    # 关键：将这批文献同样挂载到全局暗号池中，保证最后统一输出
                    idx = self.ref_pool.add(title=title, citation="", link=url)
                    count += 1
                    prog_evidence += f"\n#### [^^{idx}] {title}\n{block[:3000]}\n"
                    self.all_links_of_system.append(url)
                    
                    if count >= 10: break  # 🚨 放宽到 10 篇，专供预后 Agent 大快朵颐
            if count >= 10: break
            
        if not prog_evidence.strip():
            return "未检索到相关的专属预后文献。"
            
        logger.info(f"🧬 [Prognosis Track] 成功抓取 {count} 篇专属预后文献，已与主库物理隔离。")
        return prog_evidence
    
    
    def _reindex_references(self, content: str) -> Tuple[str, str]:
        """
        重排引用序号，并生成与 graph-ec 完全一致的参考文献格式。
        核心逻辑：根据正文实际引用的文献，在 graph-ec 的最大序号之后【连续且不跳号】地重新编号。
        """
        # 1. 匹配括号内全都是数字、逗号、空格或^符号的字符串 (如 [10], [10, 12], [^^10])
        matches_iter = re.finditer(r"\[([\d\s\^\,]+)\]", content)
        
        all_cited_ids = []
        for m in matches_iter:
            inner_text = m.group(1)
            ids = [int(s) for s in re.findall(r"\d+", inner_text)]
            if ids:
                all_cited_ids.extend(ids)

        # 2. 去重，保留正文中实际引用到的文献顺序
        unique_cited_ids = list(dict.fromkeys(all_cited_ids))
        
        old_id_to_new_id = {}
        new_references_list = []
        
        # 动态分配连续的新序号，从 graph-ec 的最大序号之后开始接力
        current_new_id = self.ref_pool.base_idx + 1 
        
        for old_id in unique_cited_ids:
            ref_obj = self.ref_pool.get_ref_by_idx(old_id)
            if ref_obj:
                old_id_to_new_id[old_id] = current_new_id
                new_references_list.append((current_new_id, ref_obj))
                current_new_id += 1
        
        # 3. 替换正文中的旧序号为新分配的连续序号
        def replace_match(match):
            inner_text = match.group(1)
            old_ids = [int(s) for s in re.findall(r"\d+", inner_text)]
            if not old_ids:
                return match.group(0)
                
            new_ids = []
            for oid in old_ids:
                if oid in old_id_to_new_id:
                    new_ids.append(str(old_id_to_new_id[oid]))
                else:
                    new_ids.append(str(oid)) 
            
            if new_ids:
                return f"[{', '.join(new_ids)}]"
            return match.group(0)
            
        new_content = re.sub(r"\[([\d\s\^\,]+)\]", replace_match, content)
        
        # 4. 生成完全模仿 graph-ec 排版格式的文本
        refs_text = "\n==================================================\n" 
        
        if new_references_list:
            for new_idx, ref in new_references_list:
                title = ref.title.replace("\n", " ").strip() if ref.title else ref.link
                if len(title) > 300: 
                    title = title[:300] + "..."
                
                pmid_val = "Unknown"
                if "pubmed.ncbi.nlm.nih.gov/" in ref.link:
                    pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', ref.link)
                    if pmid_match:
                        pmid_val = pmid_match.group(1)
                
                # 严格对齐 graph-ec 的打印格式
                if pmid_val != 'Unknown':
                    refs_text += f"[{new_idx}] PMID: {pmid_val}\n"
                else:
                    refs_text += f"[{new_idx}] URL: {ref.link[:50]}...\n"
                    
                refs_text += f"    Title: {title}\n"
                refs_text += f"    Guidelines: 前沿证据合成 (Deep Research)\n"
                refs_text += "-" * 10 + "\n"
                
        return new_content, refs_text

    async def _generate_detailed_report(
        self, current_knowledge: str, findings: List[Dict], query: str, iteration: int, prognosis_evidence: str = ""
    ):
        # 1. 恢复坚如磐石的文献标签池 (带 ^^ 暗号)
        pool_text = ""
        for i, r in enumerate(self.ref_pool.pool, 1):
            pool_text += f"[^^{self.ref_pool.base_idx + i}] {r.title}\n"

        trial_prompt = textwrap.dedent(f"""
        你是一名顶尖的妇科肿瘤循证医学分析专家。请仔细阅读以下【最新查证的前沿循证数据】，为 MDT 报告撰写《核心临床试验循证解析》部分。

        【当前患者真实病情草稿】（🚨 你的核心匹配标准！）：
        {self.treatment_context}
        
        【最新查证的前沿循证数据】：
        ---
        {current_knowledge}
        ---
        文献标签池：
        {pool_text}

        【🚨 核心任务与普适性医学逻辑红线】：
        1. **【防误杀豁免条款】**：如果文献入组标准写的是大类分期（如“纳入 III 期”），那么处于该阶段子分期的患者（如 IIIA1 期）**完全符合**入组条件！绝对不可因为带有子分期字母就误杀重磅试验！
        2. **【反幻觉绝对红线】**：你提取的任何数据必须且只能来自于上方的《前沿循证数据》原文！

        3. **【🚨 强制输出格式（必须高度还原顶级 ASCO 学术会议汇报风格）】**：
        对于保留下来的核心试验（如 PORTEC-3、GOG-0258等），必须严格按照以下格式输出。**必须精确提取原文中的 Gy 剂量、mg/m² 剂量、以及精确到小数点的生存百分比和 HR 值！如果原文未提供，则填未提供，严禁编造！**
        
        #### [填写试验名称，如 PORTEC-3] 是一项 [填写试验类型，如 多中心、随机Ⅲ期试验] [^^11]
        - **纳入人群**：[详细罗列原文要求的具体分期、组织学类型等入排标准。]
          - 💡 **入组匹配校验**：经严密核对，当前患者（[极简填写患者对应的特征]）**完全满足**该试验入组标准。
        - **分组与干预方案**：
          - [干预组A，如 放化疗组]：[必须极其详细！如 盆腔外照射 48.6 Gy/1.8 Gy/次；同步顺铂 50 mg/m²；随后卡铂 AUC 5 + 紫杉醇 175 mg/m²...]
          - [干预组B，如 单纯放疗组/单纯化疗组]：[详细的用药剂量、周期等参数]
        - **整体生存获益**：[必须包含精确数字！如 5年 OS：放化疗组 81.4% vs 单纯放疗组 76.1% (HR 0.70，p=0.034)。]
        - **分子分型与亚组分析**：[提取不同分子分型（如 p53突变）在试验中的具体获益数据。]
        - **毒性与本患者指导意义**：[提取具体毒性（如 神经病变、心血管毒性），结合本患者真实的合并症给出安全指导意见。]

        4. **【全中文与防篡改角标】**：必须直接照抄《文献标签池》中带暗号的角标（如 [^^11]）。
        
        💡 请先在 <think> 标签内进行【入组匹配校验】和【数据精准核对】，确认后再输出！
        """)
        
        followup_prompt = textwrap.dedent(f"""
        你是一名经验丰富的妇科肿瘤个案管理专家。请结合【患者初步会诊草稿】中患者的真实合并症和【最新查证的前沿循证数据】，撰写一份详尽且专业的子宫内膜癌术后随访方案。
        
        【患者初步会诊草稿】：
        {self.treatment_context}

        【最新查证的前沿循证数据】：
        {current_knowledge}

        【🚨 极其严格的格式红线】：
        1. **严禁输出任何 Markdown 标题（如 #, ##, ###, ####）**！
        2. **严禁使用中文数字大写序号（如 一、二、三）**！
        3. 你只需严格按照下方的【强制输出模板】进行内容填充。
        4. ⚠️ 内容必须体现医学专业度，请用详细、完整的医学长句进行阐述，千万不要只写简短的词组，但绝对禁止自行增加模板之外的大模块。

        【强制输出模板】（请严格原样复制以下加粗标题，在破折号后用专业医学语言详尽补充）：
        **1. 随访频率**
        - （详细说明不同时间段如前2年、3~5年、5年后的具体复查时间间隔）
        
        **2. 常规随访内容**
        - **专科查体**：（详细列出需要重点关注的全身状况评估及妇科盆腔专科检查项目，说明目的）
        - **辅助检查**：（详细列出需要定期复查的肿瘤标志物及影像学检查，如超声、MRI、胸片等，并说明不同阶段的推荐检查频率）
        
        **3. 警示症状**
        - （详细列举可能提示局部阴道穹窿复发或肺/淋巴结等远处转移的具体临床症状，并叮嘱患者出现异常时的就诊原则）
        
        **4. 生活方式与合并症管理**
        - （结合患者草稿中真实的合并症情况，给出详尽的多学科随诊建议、放化疗毒性的长程预警与管理、以及体重/代谢干预的专业指导）
        
        💡 请先在 <think> 标签内思考，确认无误后再输出上方强制模板的填空结果！
        """)

        # =====================================================================
        # 🤖 智能体 3：预后数据提取专员 (零幻觉、纯事实提取)
        # =====================================================================
        prognosis_prompt = textwrap.dedent(f"""
        你是一位严谨的肿瘤流行病学数据提取专家。
        任务：阅读以下【专为您提供的大容量预后专属文献库】，提取患者相关的预后生存率数据。

        【当前患者真实病情草稿】：
        {self.treatment_context}
        
        【专属预后文献库】（🚨 这是你唯一的数据源）：
        ---
        {prognosis_evidence}
        ---

        【🚨 提取规则与普适性医学逻辑】：
        1. **拒绝长篇大论**：不需要分析发病机制，你的唯一目标是寻找带有百分比的存活率数字（如 3年/5年/10年 OS、DFS、PFS 或中位生存期）。
        2. **【分期包容性与防漏杀机制】（最核心红线！）**：医学流行病学文献极少精确到子分期统计。如果文献提供了该患者所属的【大类分期】（例如：文献写 Stage III，而患者是 IIIA1）或【宏观类别】（如高危患者、浆液性癌总体队列）的生存率数据，你**必须**直接提取该数据，并标明这是“该分期大类/该亚型总体的生存数据”！绝对禁止因为没有精确匹配子分期就输出“未提及”！
        3. **零幻觉底线**：只有当文献库中连该患者的大分期、高危分类或特定病理（如浆液性）的数据都完全没有时，才允许输出：“专属检索文献未提及相关预后生存率数据”。
        4. **格式化输出**：输出风格必须紧凑客观，必须包含具体的百分比数据。
        5. **必留角标**：必须直接照抄上方文献库中自带的带 ^^ 的暗号（如 [^^11]），证明你的数据出处！
        
        💡 请先在 <think> 标签内运用“层级分期包容性逻辑”审视文献，找出对应大类的生存率，确认后再输出最终结果！
        """)

        async def _run_agent1():
            for attempt in range(3):
                try:
                    # 🚀 将 timeout 降至 180 秒 (3分钟)，若卡死立即重试
                    res = await invoke_with_timeout_and_retry(self.report_model, trial_prompt, timeout=180.0, max_retries=2)
                    content = remove_think_tags(res.content).strip()
                    if len(content) > 3000 or content.count("####") > 4 or "#### 4" in content:
                        logger.warning(f"⚠️ 触发护栏：Agent 1 试验数量超标 (尝试 {attempt+1}/3)，打回重做...")
                        if attempt < 2: continue
                    return content
                except Exception as e:
                    logger.warning(f"⚠️ Agent 1 执行超时或报错 (尝试 {attempt+1}/3): {e}")
                    if attempt == 2: return "临床试验数据解析生成失败，请查阅原始文献。"

        async def _run_agent15():
            for attempt in range(3):
                try:
                    res = await invoke_with_timeout_and_retry(self.report_model, followup_prompt, timeout=180.0, max_retries=2)
                    return remove_think_tags(res.content).strip()
                except Exception as e:
                    logger.warning(f"⚠️ Agent 1.5 执行超时或报错 (尝试 {attempt+1}/3): {e}")
                    if attempt == 2: return "随访方案生成失败，请参考指南常规随访。"
                
        async def _run_prognosis_agent():
            for attempt in range(3):
                try:
                    res = await invoke_with_timeout_and_retry(self.report_model, prognosis_prompt, timeout=180.0, max_retries=2)
                    return remove_think_tags(res.content).strip()
                except Exception as e:
                    logger.warning(f"⚠️ Agent 3 (预后) 执行超时或报错 (尝试 {attempt+1}/3): {e}")
                    if attempt == 2: return "暂无具体预后生存率数据。"

        logger.info("🤖 [多智能体并发] 正在执行：试验解析 & 定制随访 & 预后提取 (⏳ 已开启3分钟防卡死重试机制)...")
        
        # 🚀 给整个并发池加上最外层的强制断路器（总超时设为 10 分钟，绝对防止死锁）
        try:
            trial_analysis, followup_plan, prognosis_data = await asyncio.wait_for(
                asyncio.gather(
                    _run_agent1(), 
                    _run_agent15(), 
                    _run_prognosis_agent()
                ),
                timeout=600.0 
            )
        except Exception as e:
            logger.error(f"❌ [致命错误] 多智能体并发执行彻底超时或崩溃: {e}")
            trial_analysis = "试验解析模块超时失败。"
            followup_plan = "随访方案生成超时失败。"
            prognosis_data = "预后数据提取超时失败。"
            
        logger.info(f"📊 [预后专员提取结果]: {prognosis_data}")
        incidental_findings = self.structured_task.get("incidental_findings", [])
        incidental_str = "、".join(incidental_findings) if incidental_findings else "无"

        # =====================================================================
        # 🤖 智能体 2：子宫内膜癌 MDT 首席主笔 
        # =====================================================================
        main_prompt = textwrap.dedent(f"""
        你是一名具备顶尖国际视野的妇科肿瘤 MDT 首席专家。
        【初步会诊草稿】：
        {self.treatment_context}
        【助手整理的临床试验深度解析】：
        {trial_analysis}
        【预后专员提取的客观生存事实】（请将其作为不可辩驳的铁证直接采用！）：
        {prognosis_data}
        【翻译官拦截的次要异常】（须进行常规专科转诊）：
        {incidental_str}

        你的任务是：输出最终版的 MDT 报告主干。
        
        ## 🛑 首席专家“元临床思维”法则 (核心红线)
        1. **【循证方案的最高仲裁权】（最核心红线！）**：你绝不能做盲从草稿的“应声虫”！
           - 你的最终治疗方案**必须，且只能**建立在上方《核心临床试验循证解析》中**患者完全符合入组标准的重磅试验**基础之上！
           - 若初步草稿中的方案与上方通过了入组校验的最新试验推荐方案发生冲突，你必须以临床试验为准，果断重构主干方案！
           - **绝对不允许在方案中推荐患者根本不符合入组条件的治疗模式！**
        2. **【毒性与用药个体化纠偏】**：在敲定循证主干方案后，必须结合患者草稿中的真实合并症进行用药微调。
        3. **【全局用药一致性】**：严密比对患者的合并症，整篇报告的主干方案必须统一口径，前后逻辑自洽！
        4. **【预后数据强制量化与暗号保护】**：在【预后分析】中，必须直接自然地引用【预后专员提取的客观生存事实】中的具体数据！必须原封不动保留专员使用的带 ^^ 的文献角标（如 [^^11]）。
        5. **【合并症与次要异常的“逐条转诊”红线】**：在【术后处理】中，你必须将草稿中提到的**每一个重大合并症**和**每一个次要异常**，单独列为一个带有数字序号的项目给出专科转诊建议。
        6. **【双占位符机制】**：在试验解析部分原封不动输出 `{{{{TRIAL_PLACEHOLDER}}}}`；在随访方案部分原封不动输出 `{{{{FOLLOWUP_PLACEHOLDER}}}}`。绝对不要自己写这两个部分！

        ## 📝 必须使用的固定输出模板（将【 】内的说明替换为真实的专业分析，禁止在正文保留【 】符号！）：

        # 妇科肿瘤 MDT 最终版会诊报告

        ## 一、 病情分析
        ### 1. 病历及病理摘要
        【在此处详尽总结患者病情、高危因素及 FIGO 分期。并在末尾加上💡临床病理复核建议】

        ### 2. 各大核心指南推荐及风险分层
        - **ESGO指南风险分层**：【填写具体指南的风险分层】
        
        - **ESGO指南推荐方案**：
        `【在此处用代码块包裹原指南推荐的治疗方案。⚠️格式红线：代码块内必须提取并使用高度规格化的医学路径公式（如 "Systemic therapy ± EBRT ± VBT" 或 "Observation"），绝对禁止使用自然语言长句描述！】`
        **分析**：【在此处写出分析。如果有基于患者合并症的用药替换，必须在这里明确指出倾向于使用低毒性的现代替代方案及其理由】
        
        - **其他权威指南推荐方案**（若证据中有）：
        `【在此处用代码块包裹该指南推荐方案，同样必须使用规格化的医学路径公式格式】`
        **分析**：【在此处写出该指南的指导意见分析】
        
        ### 3. 核心临床试验循证解析
        {{{{TRIAL_PLACEHOLDER}}}}

        ## 二、 术后处理
        【请直接输出一个连续的数字列表（1、2、3...），高度还原真实医生的精简干练开医嘱风格】：
        1、 **肿瘤专科方案**：综上所述，结合上述患者符合入组标准的核心循证证据，建议患者行【填写最终敲定的放化疗方案/观察等】。建议完成治疗/术后【填写首次复查影像学的时间和具体项目】。 
        2、 **分子标志物追踪**：追踪分子分型结果（如MMR、p53等），必要时根据结果调整后续方案。
        3、 **【合并症/次要异常 1 名称，如：冠心病】**：建议转诊至【对应科室】门诊进一步评估与随诊，【给出简明建议，如评估放化疗风险】。
        4、 **【合并症/次要异常 2 名称，如：浅表性胃炎】**：建议转诊至消化内科门诊进一步评估与处理。
        【⚠️ 请继续用 5、6、7... 顺延列出患者所有的重大合并症和次要异常，每一个异常必须单独占一条！若全部罗列完毕则结束该段。】

        ## 三、 预后分析
        【结合患者特有高危因素，将上方预后专员提取的数据极其精简干练地串联进来。如果专员反馈“暂无数据”，则使用指南通用的评估话术。如果有详细的数据必须保留！】

        ## 四、 随访方案
        {{{{FOLLOWUP_PLACEHOLDER}}}}
        
        💡 请先在 <think> 标签内思考确认无误后再输出正文！
        """)  

        logger.info("🤖 [Agent 2] 正在统筹生成 MDT 报告主干并确保全局逻辑一致性...")
        
        max_guardrail_retries = 3
        content = ""
        
        for attempt in range(max_guardrail_retries):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, main_prompt, timeout=1200.0, max_retries=3
                )
                content = remove_think_tags(response.content)
                
                banned_phrases = ["代码块包裹", "在此处写出", "严禁遗漏", "医学路径公式", "【", "】"]
                lazy_generation_detected = any(phrase in content for phrase in banned_phrases)
                
                if lazy_generation_detected:
                    logger.warning(f"⚠️ 触发护栏：Agent 2 照抄指令或遗留模板括号 (尝试 {attempt + 1}/{max_guardrail_retries})，打回重做...")
                    if attempt < max_guardrail_retries - 1: continue
                    else: logger.error("❌ 达到最大重试次数，强制继续。")
                
                for cut_word in ["## 五", "# 五", "参考文献", "References"]:
                    if cut_word in content:
                        content = content.split(cut_word)[0].strip()
                break
                
            except Exception as e:
                logger.error(f"Agent 2 生成报错: {e}")
                if attempt == max_guardrail_retries - 1:
                    content = "报告生成失败"

        try:
            if "{{TRIAL_PLACEHOLDER}}" in content:
                content = content.replace("{{TRIAL_PLACEHOLDER}}", trial_analysis)
            else:
                if "## 二、 术后处理" in content:
                    content = content.replace("## 二、 术后处理", f"### 3. 核心临床试验循证解析\n{trial_analysis}\n\n## 二、 术后处理")
                else:
                    content += f"\n\n### 3. 核心临床试验循证解析\n{trial_analysis}"
                    
            if "{{FOLLOWUP_PLACEHOLDER}}" in content:
                content = content.replace("{{FOLLOWUP_PLACEHOLDER}}", followup_plan)
            else:
                if "## 四、 随访方案" in content:
                    content = content.replace("## 四、 随访方案", f"## 四、 随访方案\n{followup_plan}\n")
                else:
                    content += f"\n\n## 四、 随访方案\n{followup_plan}"

            new_content, refs_section = self._reindex_references(content)
            full_report = new_content + "\n" + refs_section
            return full_report, full_report 
            
        except Exception as e:
            logger.error(f"Failed to post-process references: {e}")
            fallback_refs = "\n==================================================\n"
            for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                fallback_refs += f"[{i}] URL: {ref.link}\n    Title: {ref.title or ref.link}\n----------\n"
            return "处理失败", "处理失败"
        
        
    async def _extract_knowledge(self, facts_md: str, refs_in_round: List[Dict]):
        """Extract key info from tool outputs."""
        prompt = f"""
        你是一名顶尖的临床数据挖掘专家，请从以下检索结果中提取具有绝对临床级精度的医学事实。
        
        【🚨 数据保留致命红线】：
        1. 必须原封不动地保留所有精确数字：药物具体剂量（如 mg/m2, AUC）、放疗参数（如 Gy/次）、周期数。
        2. 必须保留所有统计学硬核数据：具体生存期百分比（OS/DFS/RFS等）、风险比(HR)、95%置信区间(CI)及 P值。
        3. 必须高度关注并保留亚组分析（Subgroup Analysis）和分子分型（如 p53abn, POLE 等）的差异化数据。
        绝对禁止做笼统的语义概括（如“疗效更好”、“显著改善”等废话）！
        
        Facts:
        {facts_md}
        
        Refs:
        {json.dumps(refs_in_round)}
        
        Output JSON:
        {{
            "key_information": "- **[试验名称/主题]**：具体干预方案细节... OS为xx% (HR=xx, 95%CI: xx-xx, P=xx)... 亚组分析显示... (<url>)",
            "cleaned_refs": [{{"url": "...", "description": "Title/Summary"}}]
        }}
        """
        try:
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=1200.0)
            cleaned_content = remove_think_tags(resp.content)
            data = safe_json_from_text(cleaned_content) or {}
            return data.get("key_information", ""), data.get("cleaned_refs", [])
        except Exception:
            return facts_md, refs_in_round
        

    async def process_multiple_knowledge_chunks(self, query: str, current_key_info: str) -> str:
        """Consolidate knowledge."""
        if not self.knowledge_chunks:
            return current_key_info
        
        prompt = f"""
        你是一名高级循证医学专家。请将以下医学检索发现整合为一份极其硬核、详细的临床摘要。
        
        【🚨 致命红线：禁止“洗稿”和概括】：
        你绝对不能像写新闻报道一样概括这些数据！必须 100% 继承并罗列原本存在的：
        - 试验名称、NCT号及期数（Phase III）
        - 具体入组条件与随机分组策略
        - 用药方案的精确剂量和给药频率
        - OS/DFS/PFS的具体百分比、HR值、95%CI 和 P值
        - 基于分子分型（如 p53abn 等）的特定亚组统计数据
        请在每一句话后保留引用的 <URL>。
        
        Findings:
        {current_key_info}
        """
        try:
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=1200.0)
            return remove_think_tags(resp.content)
        except Exception:
            return current_key_info

    async def _extract_structured_data(self, raw_text: str, source_type: str, query: str) -> str:
        """针对 ClinicalTrials 和 FDA 的超长 JSON 数据进行精准提纯压缩"""
        if source_type == "clinicaltrials":
            prompt = textwrap.dedent(f"""
            你是一名顶尖的临床试验数据提取专家。
            以下是来自 ClinicalTrials.gov 的原始 JSON 数据。它包含了大量的冗余字段，请你像“数据榨汁机”一样，提取出最核心的临床信息。
            
            【目标查证核心】：{query}
            
            【🚨 提取红线（必须包含精确数字）】：
            1. **基本信息**：提取试验名称、NCT编号、试验分期。
            2. **入组标准**：简述核心的纳入/排除标准。
            3. **干预方案**：精确提取用药名称、剂量（如 mg/m2, AUC）、放疗参数（Gy）和给药周期。绝对不能遗漏数字！
            4. **核心结果（若有）**：精确提取 OS, DFS, PFS 的百分比，HR 值，95% CI 和 P 值。
            5. **安全性**：提取相关的严重毒副作用及发生率。
            
            💡 思考后，请直接输出高度浓缩的中文医学总结。如果找不到某项数据，直接忽略即可。
            
            Raw Data (Truncated):
            {raw_text[:20000]} 
            """)
        else: # FDA
            prompt = textwrap.dedent(f"""
            你是一名专业的 FDA 药品说明书解析专家。
            以下是 FDA 数据库返回的药品说明书 JSON 数据。请针对目标问题，提取最关键的安全性与用药指导信息。
            
            【目标查证核心】：{query}
            
            【🚨 提取红线】：
            1. **药品身份**：提取药品名称（Brand name / Generic name）。
            2. **黑框警告与毒性**：提取黑框警告、重点提取与【心血管】或【神经系统】相关的严重毒副反应及管理建议。
            3. **特殊人群**：提取针对高龄、合并症（如糖尿病、高血压、脑梗）患者的剂量调整建议或禁忌症。
            
            💡 思考后，请直接输出高度浓缩的中文医学总结，保留所有指导性数字和数据。
            
            Raw Data (Truncated):
            {raw_text[:20000]}
            """)

        try:
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=800.0)
            return remove_think_tags(resp.content).strip()
        except Exception as e:
            logger.error(f"FDA/CT Extraction failed: {e}")
            return "数据提取失败，未发现有效量化信息。"
    
    
    async def analyze_topic(self, query: str) -> Dict:
        """Main execution loop (终极并发版：局部沙盒隔离 + 暴力脱壳 + 全局去重)."""
        logger.info(f"Starting Pure API Validation (High-Performance Concurrent Mode)")
        
        current_knowledge = ""
        cumulative_raw_evidence = "" 
        iteration = 0
        findings = []

        # 🚀 全局跨轮去重记忆池
        global_seen_urls = set()

        # =====================================================================
        # 🚦 核心并发控制阀门 (Semaphore)
        # =====================================================================
        llm_semaphore, api_semaphore = get_global_semaphores()

        await self.initialize()

        # 🚀 [新增] 启动后台独立的预后检索管线，不阻塞主流程！
        prognosis_task = asyncio.create_task(self._run_prognosis_retrieval_track())

        while iteration < self.max_iterations:
            questions = await self._get_follow_up_questions(current_knowledge, query)
            if not questions:
                questions = [query]
                
            self.questions_by_iteration[iteration] = questions
            logger.info(f"🚀 Iteration {iteration+1}: Concurrently processing {len(questions)} sub-questions...")
            
            async def fetch_and_parse_question(q: str):
                logger.info(f"🔍 [Task Started] {q}")
                
                local_selector = ToolSelector(
                    self.tool_planning_model,
                    self.reasoning_model,
                    self.mcp_tool_client,
                    tool_info_data=None,
                    embedding_api_key=None,
                    embedding_cache=None,
                    available_tools=self.chosen_tools,
                )
                local_executor = ToolExecutor(
                    self.mcp_tool_client, self.error_log_path, self.fast_model
                )

                try:
                    async with llm_semaphore:
                        t_calls = await local_selector.run(q)
                except Exception as e:
                    logger.warning(f"Tool selection failed for {q}: {e}")
                    t_calls = []

                if not t_calls:
                    return []

                for call in t_calls:
                    if 'tool_input' in call and 'query' in call['tool_input']:
                        raw_q = str(call['tool_input']['query'])
                        call['tool_input']['query'] = raw_q.replace("'", '"')

                try:
                    async with api_semaphore:
                        t_results = await local_executor.run(t_calls) or []
                        
                    try:
                        with open("API_RAW_OUTPUT_CONCURRENT.txt", "a", encoding="utf-8") as f:
                            f.write(f"\n{'='*60}\n🔍 并发检索词: {q}\n{'='*60}\n")
                            for idx, res in enumerate(t_results):
                                f.write(f"--- 来源片段 {idx+1} ---\n{str(res)}\n\n")
                    except Exception:
                        pass
                        
                except Exception as e:
                    logger.error(f"Tool execution failed for '{q}': {e}")
                    t_results = []
                
                return t_results

            logger.info(f"⚡ 启动并发检索 (发射 {len(questions)} 个独立沙盒探索分支)...")
            all_questions_results = await asyncio.gather(
                *(fetch_and_parse_question(q) for q in questions)
            )

            unique_articles_dict = {}
            for tool_results in all_questions_results:
                if not tool_results: continue
                
                for res in tool_results:
                    res_str = ""
                    if isinstance(res, dict) and "content" in res:
                        raw_content = res["content"]
                        try:
                            import ast
                            parsed_list = ast.literal_eval(raw_content)
                            if isinstance(parsed_list, list):
                                res_str = "".join([item.get("text", "") for item in parsed_list])
                        except Exception:
                            res_str = str(raw_content)
                    else:
                        res_str = str(res)
                        
                    res_str = res_str.replace('\\n', '\n')
                    
                    res_str = re.sub(r"^\[?\s*\{\s*['\"]type['\"]\s*:\s*['\"]text['\"]\s*,\s*['\"]text['\"]\s*:\s*['\"]", "", res_str)
                    res_str = re.sub(r"['\"]\s*\}\s*\]?$", "", res_str)
                        
                    blocks = res_str.split("\n---\n") if "\n---\n" in res_str else [res_str]
                    
                    for block in blocks:
                        if not block.strip(): continue
                        
                        url = ""
                        pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', block, re.IGNORECASE) or re.search(r'["\']?(?:PMID|uid|id)["\']?\s*[:=]\s*["\']?(\d{7,9})["\']?', block, re.IGNORECASE)
                        nct_match = re.search(r'(NCT\d{8})', block, re.IGNORECASE)
                        
                        if pmid_match:
                            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
                        elif nct_match:
                            url = f"https://clinicaltrials.gov/study/{nct_match.group(1)}"
                        elif "openfda" in block.lower() or "brand_name" in block.lower() or "generic_name" in block.lower():
                            url = "https://nctr-crs.fda.gov/fdalabel/ui/search"
                            
                        if not url: continue
                        
                        title = "Unknown Title"
                        title_match = re.search(r'^(?:Article )?Title:\s*([^\n]+)', block, re.IGNORECASE | re.MULTILINE)
                        if not title_match:
                            title_match = re.search(r'\bTitle:\s*([^\n]+)', block, re.IGNORECASE)
                        if not title_match:
                            title_match = re.search(r'"title"\s*:\s*"([^"]+)"', block, re.IGNORECASE) or re.search(r'"BriefTitle"\s*:\s*"([^"]+)"', block, re.IGNORECASE)
                        if title_match:
                            title = title_match.group(1).strip()
                            
                        if len(title) < 15 and "FDA" not in title and "Unknown" not in title:
                            continue
                            
                        if url not in global_seen_urls:
                            global_seen_urls.add(url) 
                            
                            raw_text = block.strip()
                            if len(raw_text) > 6000:
                                raw_text = raw_text[:6000] + "\n\n...[文本过长，为防止 Token 溢出已执行物理截断]..."
                            
                            unique_articles_dict[url] = {
                                "url": url,
                                "title": title,
                                "content": raw_text
                            }

            keys_to_extract = []
            for url, art in unique_articles_dict.items():
                if "clinicaltrials.gov" in url:
                    keys_to_extract.append((url, "clinicaltrials"))
                elif "fda.gov" in url or "nctr-crs.fda.gov" in url:
                    keys_to_extract.append((url, "fda"))

            async def process_extraction_safely(url, s_type):
                async with llm_semaphore:
                    raw = unique_articles_dict[url]["content"]
                    extracted = await self._extract_structured_data(raw, s_type, query)
                    unique_articles_dict[url]["content"] = extracted
                
            if keys_to_extract:
                logger.info(f"🧬 [Data Refinery] 对 {len(keys_to_extract)} 篇结构化数据启动受控并发提纯...")
                await asyncio.gather(*(process_extraction_safely(u, t) for u, t in keys_to_extract))

            articles_list = list(unique_articles_dict.values())
            selected_articles = []
            
            if len(articles_list) > 5:
                logger.info(f"🔍 [Screening] 启动大模型初筛机制，评估 {len(articles_list)} 篇文献/数据...")
                
                titles_catalog = ""
                for idx, art in enumerate(articles_list):
                    # 🚀 [核心修改]：在菜单中为 CT 和 FDA 打上强力高光标签！
                    if "clinicaltrials.gov" in art["url"]:
                        titles_catalog += f"[{idx}] 🏥 [专属结构化提纯 - 临床试验 NCT] {art['title']}\n"
                    elif "fda.gov" in art["url"] or "nctr-crs.fda.gov" in art["url"]:
                        titles_catalog += f"[{idx}] 💊 [专属结构化提纯 - FDA 药物数据] {art['title']}\n"
                    else:
                        titles_catalog += f"[{idx}] 📄 [PubMed 前沿文献] {art['title']}\n"
                
                screening_prompt = textwrap.dedent(f"""
                你是一名顶尖的妇科肿瘤循证医学文献筛选专家。
                我们为患者检索并初步结构化了以下 {len(articles_list)} 篇候选文献/试验数据。
                为了防止信息过载，请你挑选出最核心、最具有指导意义的 1 到 5 篇。
                
                【患者病情与检索背景】：
                {self.treatment_context}
                
                【候选数据菜单】：
                {titles_catalog}
                
                【🚨 筛选红线（极度重要！）】：
                1. **禁止“以貌取人”**：带有 [专属结构化提纯] 标签的 NCT 或 FDA 数据，虽然标题可能枯燥，但我们已经在后台将其提取为高价值的干预参数！你必须认真评估其与当前患者病情的匹配度，**若高度相关，请赋予其最高优先级入选**！
                2. **严格防噪（宁缺毋滥）**：如果某项 NCT 试验或 PubMed 文献的入组人群（如分期、分子分型）与当前患者**明显不符**，即使它是重磅试验，也必须**果断剔除**！绝对不要为了凑数而选入不相关的数据！
                3. 对于 [PubMed 前沿文献]，优先选择重磅临床试验（如 PORTEC 系列等）的大样本长期随访结果。
                4. 坚决剔除垃圾匹配（如普拉提、骨科等无关领域）和基础动物实验。
                
                【强制输出格式】：
                请严格输出一个 JSON 数组，包含你选中的文献编号（最多选5个！绝对不要输出除了 JSON 数组外的多余正文！）。
                例如：[0, 2, 4]
                
                💡 请先在 <think> 标签内进行充分的医学甄别思考，重点衡量各数据源的含金量与患者匹配度，确认无误后再输出 JSON 数组！
                """)
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        resp = await invoke_with_timeout_and_retry(self.model, screening_prompt, timeout=800.0)
                        cleaned_resp = remove_think_tags(resp.content)
                        json_match = re.search(r'\[[\d\s,]+\]', cleaned_resp)
                        
                        if json_match:
                            selected_indices = json.loads(json_match.group(0))
                            valid_indices = [i for i in set(selected_indices) if isinstance(i, int) and 0 <= i < len(articles_list)]
                            
                            if valid_indices:
                                selected_articles = [articles_list[i] for i in valid_indices]
                                logger.info(f"✅ [Screening] 成功提取到 {len(selected_articles)} 篇文献。")
                                break
                            else:
                                logger.warning(f"⚠️ [Screening] 提取到的有效文献序号为空 (尝试 {attempt+1}/{max_retries})...")
                        else:
                            logger.warning(f"⚠️ [Screening] 模型未按格式输出 JSON 数组 (尝试 {attempt+1}/{max_retries})...")
                    except Exception as e:
                        logger.warning(f"⚠️ [Screening] 标题初筛执行报错 (尝试 {attempt+1}/{max_retries}): {e}")
                    
                if not selected_articles:
                    logger.error("❌ 达到最大重试次数，退回默认选取前 5 篇。")
                    selected_articles = articles_list[:5]
            else:
                selected_articles = articles_list
                
            logger.info(f"🔍 [Screening] 最终挑选了 {len(selected_articles)} 篇【最高价值】文献进入直通车。")

            chunk_evidence = f"\n\n### 第 {iteration + 1} 轮筛选出的核心医学证据：\n"
            
            for art in selected_articles:
                ref_url = art["url"]
                ref_title = art["title"]
                raw_content = art["content"]
                
                idx = self.ref_pool.add(title=ref_title, citation="", link=ref_url)
                
                chunk_evidence += f"\n#### [^^{idx}] {ref_title}\n"
                chunk_evidence += f"{raw_content}\n"
                
                self.all_links_of_system.append(ref_url)

            cumulative_raw_evidence += chunk_evidence

            current_knowledge = await self._answer_query(
                cumulative_raw_evidence, query, iteration, self.max_iterations
            )
            iteration += 1

        # 🚀 [新增] 等待后台预后检索管线完成，并获取专属预后文献
        prognosis_evidence = await prognosis_task

        final_report = ""
        if self.is_report:
            try:
                # 🚀 [修改] 将 prognosis_evidence 传给详尽报告生成器
                final_report_tuple = await self._generate_detailed_report(
                    cumulative_raw_evidence, findings, query, iteration, prognosis_evidence
                )
                if isinstance(final_report_tuple, tuple):
                    final_report = final_report_tuple[1]
                else:
                    final_report = str(final_report_tuple)
            except Exception as e:
                logger.warning(f"Failed to generate detailed report: {e}")
                fallback_refs = "\n==================================================\n"
                for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                    title = ref.title if ref.title else "Source"
                    fallback_refs += f"[{i}] URL: {ref.link}\n    Title: {title}\n----------\n"
                final_report = current_knowledge + fallback_refs

        return {
            "findings": findings,
            "iterations": iteration,
            "questions": self.questions_by_iteration,
            "current_knowledge": current_knowledge,
            "final_report": final_report,
        }
