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

# Import utilities
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


def remove_think_tags(text: str) -> str:
    """
    Robustly remove <think> tags from model output.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned:
        cleaned = re.sub(r".*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class ReferencePool:
    """Reference pool for citations."""
    def __init__(self) -> None:
        self.pool: List[SourcesReference] = []
        self.link2idx: dict[str, int] = {}

    def add(self, title: str, citation: str, link: str) -> int:
        if not link:
            return -1
        if link in self.link2idx:
            return self.link2idx[link]
        idx = len(self.pool) + 1
        self.link2idx[link] = idx
        self.pool.append(
            SourcesReference(title=title or link, subtitle=citation or "", link=link)
        )
        return idx
    
    def get_ref_by_idx(self, idx: int):
        if 1 <= idx <= len(self.pool):
            return self.pool[idx-1]
        return None


class AdvancedSearchSystem:
    def __init__(
        self,
        max_iterations=2,
        questions_per_iteration=4,
        is_report=True,
        chosen_tools: list[str] = None,
        error_log_path: str = "",
        using_model = "deepseek",  
        treatment_context: str = "",
    ):
        self.ref_pool = ReferencePool() 
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
                local_llm = get_local_model(temperature=0.5)
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
            # Default to GPT
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

    async def _get_follow_up_questions(
        self, current_knowledge: str, query: str
    ) -> List[str]:
        """
        Generate API-compatible search queries (keywords).
        """
        now = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        # Clinical Evidence Validator (Pure API Mode)
        
        ## Context
        - Task: Validate the provided treatment plan against LATEST (2024-Present) official evidence.
        - Date: {now}
        - Input Plan/Context: "{self.treatment_context[:2000]}..."
        - Current Knowledge: {current_knowledge}
        
        ## Your Task
        Generate exactly {self.questions_per_iteration} specific API queries to verify or formulate the plan.
        
        ## STRICT QUERY RULES (For APIs)
        1. **PubMed**: Use **KEYWORDS ONLY**. No "site:" operator.
           - Bad: "Search for papers about Pembrolizumab survival"
           - Good: "Pembrolizumab endometrial cancer overall survival 2024"
        2. **ClinicalTrials.gov**: Use **Condition | Intervention** format.
           - Bad: "Find trials for..."
           - Good: "Endometrial Cancer | Lenvatinib"
        3. **FDA**: Use **Drug Name** only.
           - Bad: "Is Dostarlimab approved?"
           - Good: "Dostarlimab"
        
        ## Output Format (JSON)
        {{
            "thoughts": "Checking standard of care for Stage III.",
            "strategy": ["Check NCCN via PubMed", "Check Active Trials"],
            "sub_queries": [
                "Endometrial cancer stage III adjuvant therapy 2024", 
                "Endometrial Cancer | Chemotherapy"
            ]
        }}
        """

        try:
            response = await invoke_with_timeout_and_retry(
                self.reasoning_model, prompt, timeout=1200.0, max_retries=3
            )
            
            response_text = remove_think_tags(response.content)
            
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = safe_json_from_text(response_text)
                if parsed:
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
        """
        Synthesize API findings.
        """
        existing_refs = [
            f"[{idx}] {ref.link} — {ref.title}"
            for idx, ref in enumerate(self.ref_pool.pool, 1)
        ]
        refs_block = "\n".join(existing_refs) or "*None yet*"

        prompt = textwrap.dedent(f"""
        ## Task: Evidence Synthesis
        
        Validating clinical plan using **OFFICIAL API DATA** (PubMed/FDA/CT.gov).
        
        ## Input Plan
        {self.treatment_context}
        
        ## Verified API Data
        {current_knowledge}
        
        ## Sources
        {refs_block}
        
        ## Instructions
        1. **Validate**: Does 2024-Present evidence support the plan?
        2. **Update**: Identify newer trials/approvals.
        3. **Detail**: Note specific regimens (Drug/Dose) found in evidence.
        
        ## Output Template
        ## Evidence Status
        - **Decision Point**: [e.g. Adjuvant Therapy]
          - **Status**: [✅ Supported / ⚠️ Controversy / ❓ No Data]
          - **Key Evidence**: [Summarize findings from PubMed/CT.gov [^^n]]
        
        ## Missing Data
        [What couldn't be verified?]

        ## References
        [List [^^n] citations strictly]
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

    def _reindex_references(self, content: str) -> Tuple[str, str]:
        """
        Extract citations from text, supporting various formats like [^^1], [^^1, ^^2], [^^1, 2].
        Reindex them starting from 1 and generate a clean reference list.
        """
        # 1. 更加通用的正则：捕获 [ ... ] 中包含 ^^ 的内容
        # 允许 [^^1], [^^1, ^^2], [^^1, 2], [^^1,2]
        matches_iter = re.finditer(r"\[(.*?)\]", content)
        
        all_cited_ids = []
        valid_matches = [] # 存储合法的引用对象，用于后续替换
        
        for m in matches_iter:
            inner_text = m.group(1)
            # 只有当括号内包含 '^^' 时才认为是引用标记
            if "^^" in inner_text:
                # 提取里面的所有数字，忽略 ^^, 逗号, 空格
                # 例如 "doc ^^4, ^^9" -> [4, 9]
                ids = [int(s) for s in re.findall(r"\d+", inner_text)]
                if ids:
                    all_cited_ids.extend(ids)
                    valid_matches.append(m.group(0)) # 保存原始字符串用于正则替换

        # 2. 建立映射：原始ID -> 新序号 (保持出现顺序且去重)
        unique_cited_ids = list(dict.fromkeys(all_cited_ids))
        
        old_id_to_new_id = {}
        new_references_list = []
        current_new_id = 1
        
        for old_id in unique_cited_ids:
            ref_obj = self.ref_pool.get_ref_by_idx(old_id)
            if ref_obj:
                old_id_to_new_id[old_id] = current_new_id
                new_references_list.append(ref_obj)
                current_new_id += 1
        
        # 3. 替换正文中的引用
        def replace_match(match):
            inner_text = match.group(1)
            if "^^" not in inner_text:
                return match.group(0) # 不是我们的引用格式，原样返回
            
            # 提取数字
            old_ids = [int(s) for s in re.findall(r"\d+", inner_text)]
            
            new_ids_str = ""
            for oid in old_ids:
                if oid in old_id_to_new_id:
                    # 使用临时标记 [^^^n] 防止多次替换冲突
                    new_ids_str += f"[^^^{old_id_to_new_id[oid]}]"
            
            return new_ids_str if new_ids_str else match.group(0)
            
        # 使用更宽泛的正则进行替换
        new_content = re.sub(r"\[(.*?)\]", replace_match, content)
        new_content = new_content.replace("[^^^", "[")
        
        # 4. 生成参考文献列表文本
        refs_text = "\n\n**六、参考文献 (References)**\n"
        if not new_references_list:
            refs_text += "*No references cited in the report.*\n"
        else:
            for i, ref in enumerate(new_references_list, 1):
                title = ref.title.replace("\n", " ").strip() if ref.title else ref.link
                if len(title) > 300: 
                    title = title[:300] + "..."
                # 生成 Markdown 链接格式 [1] [Title](Link)
                refs_text += f"[{i}] [{title}]({ref.link})\n"
                
        return new_content, refs_text

    async def _generate_detailed_report(
        self, current_knowledge: str, findings: List[Dict], query: str, iteration: int
    ):
        """
        Generate the final report with detailed clinical recommendations.
        """
        pool_objs = [
            {"idx": i, "url": r.link, "desc": r.title} 
            for i, r in enumerate(self.ref_pool.pool, 1)
        ]
        pool_json = json.dumps(pool_objs, indent=2)

        prompt = textwrap.dedent(f"""
        你是一名高级临床决策支持系统。请基于提供的【患者病史/拟定方案】和【最新API检索证据】，生成一份详尽的**临床治疗建议书**。
        
        ### 患者上下文 (Context)
        {self.treatment_context}
        
        ### 2024-Present 最新证据 (Evidence)
        {current_knowledge}
        
        ### 引用池 (Citation Pool)
        ```json
        {pool_json}
        ```

        ## 任务要求 (Requirements)
        1. **格式**：严格模仿正规临床病历/Tumor Board报告的后续部分。
        2. **语言**：简体中文。
        3. **详细程度**：**非常详细**。不要只写“建议化疗”，必须写出具体方案（如：药物名称、推荐周期数、给药方式）。对于放疗，指明靶区范围。
        4. **证据支撑**：所有建议必须引用检索到的证据[^^n]。
        5. **参考文献**：你可以在正文中使用[^^n]，但**不需要**在文末生成Reference列表，系统会自动追加。

        ### 输出模版 (Output Template)

        **一、术后辅助治疗建议 (Post-operative Adjuvant Therapy Recommendations)**
        1. **全身治疗 (Systemic Therapy)**：
           - *推荐方案*：[详细写出药物，如：紫杉醇 175mg/m² + 卡铂 AUC 5-6]
           - *周期*：[如：每3周一次，共6个周期]
           - *依据*：[简述为何选择此方案，引用 ^^n]
        2. **放射治疗 (Radiation Therapy)**：
           - *推荐术式*：[如：盆腔外照射 (EBRT) ± 阴道近距离放疗]
           - *靶区与剂量*：[如：45-50.4 Gy / 25-28 fx]
           - *争议点说明*：[如：阴道近距离放疗是否必要，引用 ^^n]

        **二、循证医学证据分析 (Evidence-based Rationale)**
        *结合患者的分子分型（如 p53/MMR 状态）和 FIGO 分期，深入分析为何该方案是2024-2025年的最佳选择。讨论相关的重要临床试验（如 PORTEC-3, GOG-258 等）结果支持。*

        **三、风险评估与管理 (Risk Assessment & Management)**
        *详细列出可能的不良反应（血液学毒性、神经毒性、消化道反应）及其预防和处理措施。*

        **四、随访计划 (Follow-up Plan)**
        *列出具体的时间表（如：术后2年内每3个月一次...）及每次随访的必查项目（体格检查、影像学、肿瘤标志物等）。*
        
        **五、前沿进展与临床试验 (Emerging Updates & Trials)**
        *若有适合该患者的入组机会（NCT...）或新药（如免疫检查点抑制剂）的适应症更新，在此列出。*
        """)

        try:
            # 2. 尝试生成报告正文
            response = await invoke_with_timeout_and_retry(
                self.report_model, prompt, timeout=1200.0, max_retries=3
            )
            content = remove_think_tags(response.content)
            
            # 3. ✅ [重排引用] 调用新的重排函数
            new_content, refs_section = self._reindex_references(content)
            
            full_report = new_content + refs_section
            return full_report, full_report 
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            
            # 兜底：列出所有文献，防止数据丢失
            fallback_refs = "\n\n**六、参考文献 (References)**\n"
            for i, ref in enumerate(self.ref_pool.pool, 1):
                title = ref.title if ref.title else ref.link
                fallback_refs += f"[{i}] [{title}]({ref.link})\n"
            
            final_report = f"## 报告生成失败 (Report Generation Failed)\n\n{current_knowledge}\n" + fallback_refs
            return final_report, final_report

    async def _extract_knowledge(self, facts_md: str, refs_in_round: List[Dict]):
        """Extract key info from tool outputs."""
        prompt = f"""
        Extract key clinical facts from these API results.
        
        Facts:
        {facts_md}
        
        Refs:
        {json.dumps(refs_in_round)}
        
        Output JSON:
        {{
            "key_information": "- **Fact**... (<url>)",
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
        Consolidate these research findings into a concise summary.
        Keep citations <URL>.
        
        Findings:
        {current_key_info}
        """
        try:
            resp = await invoke_with_timeout_and_retry(self.fast_model, prompt, timeout=1200.0)
            return remove_think_tags(resp.content)
        except Exception:
            return current_key_info

    async def analyze_topic(self, query: str) -> Dict:
        """Main execution loop."""
        logger.info(f"Starting Pure API Validation for plan length: {len(self.treatment_context)}")
        
        current_knowledge = ""
        iteration = 0
        findings = []

        await self.initialize()

        while iteration < self.max_iterations:
            questions = await self._get_follow_up_questions(current_knowledge, query)
            if not questions:
                questions = [query]
                
            self.questions_by_iteration[iteration] = questions
            
            fullquery_tool_results = []
            
            for question in questions:
                try:
                    tool_calls = await self.tool_selector.run(question)
                except Exception as e:
                    logger.warning(f"Tool selection failed: {e}")
                    tool_calls = []

                if not tool_calls:
                    continue
                
                # ✅ [新增] 强制注入 ClinicalTrials 的时间过滤器
                for call in tool_calls:
                    if call.get("tool_name") in ["get_studies", "clinicaltrials_search"]:
                        inputs = call.get("tool_input", {})
                        original_q = inputs.get("query", "")
                        # 确保只搜索 2024 年至今有过更新的试验
                        date_constraint = " AND AREA[LastUpdatePostDate]RANGE[01/01/2024, MAX]"
                        
                        if "LastUpdatePostDate" not in original_q:
                            inputs["query"] = f"{original_q}{date_constraint}"
                            logger.info(f"Injecting date filter to ClinicalTrials query: {inputs['query']}")

                try:
                    tool_results = await self.tool_executor.run(tool_calls) or []
                except Exception as e:
                    logger.error(f"Tool execution failed for question '{question}': {e}")
                    tool_results = []
                
                if tool_results:
                    try:
                        parsed_list = await asyncio.gather(
                            *(parse_single(res, query=question) for res in tool_results)
                        )
                        compressed = await compress_all_llm(
                            self.fast_model, parsed_list, limit=3, query=query
                        )
                        fullquery_tool_results.extend(compressed)
                    except Exception as e:
                        logger.warning(f"Error parsing tool results: {e}")

            iteration += 1
            facts, refs_raw = [], []
            for item in fullquery_tool_results:
                facts.extend(item.get("extracted_facts", []))
                refs_raw.extend(item.get("references", []))

            unique_refs = {}
            for ref in refs_raw:
                url = ref.get("url", "").strip()
                if url and url.startswith("http") and url not in unique_refs:
                    unique_refs[url] = ref
            refs = list(unique_refs.values())
            self.all_links_of_system.extend([r["url"] for r in refs])

            facts_md = "\n".join(f"- {f}" for f in facts)
            key_info, cleaned_refs = await self._extract_knowledge(facts_md, refs)
            
            self.knowledge_chunks.append({"key_info": key_info})
            current_knowledge = await self.process_multiple_knowledge_chunks(query, key_info)

            for ref in cleaned_refs:
                idx = self.ref_pool.add(
                    title=ref.get("description", ref["url"]),
                    citation="",
                    link=ref["url"]
                )
                current_knowledge = current_knowledge.replace(ref["url"], f"[{idx}]")

            final_answer = await self._answer_query(
                current_knowledge, query, iteration, self.max_iterations
            )
            current_knowledge = final_answer

        final_report = ""
        if self.is_report:
            try:
                final_report_tuple = await self._generate_detailed_report(
                    current_knowledge, findings, query, iteration
                )
                if isinstance(final_report_tuple, tuple):
                    final_report = final_report_tuple[1]
                else:
                    final_report = str(final_report_tuple)
            except Exception as e:
                logger.warning(f"Failed to generate detailed report: {e}")
                
                # 兜底：列出所有文献，防止数据丢失
                fallback_refs = "\n\n**六、参考文献 (References)**\n"
                for i, ref in enumerate(self.ref_pool.pool, 1):
                    title = ref.title if ref.title else "Source"
                    fallback_refs += f"[{i}] [{title}]({ref.link})\n"
                final_report = current_knowledge + fallback_refs

        return {
            "findings": findings,
            "iterations": iteration,
            "questions": self.questions_by_iteration,
            "current_knowledge": current_knowledge,
            "final_report": final_report,
        }