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

        1. **【灯塔试验复核与预后长期随访（占用 2-4 个额度）】**：
           - 针对上游报告中提及的治疗方案，检索支撑该方案的最核心临床试验（如 PORTEC, GOG, NRG 系列等）的最新随访生存数据。
           - ⚠️ 检索词示例：PORTEC-3 AND ("survival"[Title/Abstract] OR "outcomes"[Title/Abstract]) AND 2018:2026[dp]
        2. **【高价值 PICO 问题精准查证（占用 2-4 个额度）】**：
           - 提取结构化数据中遗留的 PICO 问题，转化为具体的 PubMed 检索词。重点关注特定分子分型（如 NSMP, p53abn）对预后的细分影响或前瞻性证据。
           - ⚠️ 检索词示例：Endometrial cancer AND NSMP AND "recurrence risk" AND 2020:2026[dp]
        （注：上游提供的初步方案已通过严格的指南毒理审核，**你绝对不需要**再生成验证药物毒性、寻找卡铂/顺铂替代方案等纠偏性质的检索词！）

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
        请根据以下患者信息，生成一个【极高命中率】的预后生存率大样本数据检索词（Boolean string）。
        
        【患者信息】：
        {base_info}
        
        【🚨 严苛规则（极度重要）】：
        1. **核心要素**：你的检索词只能包含：疾病大类名称 + 核心病理类型 + 分期 + 数据库限定词。
        2. **【分期降维】（最核心红线！）**：由于医学文献极少精确到子分期，如果患者的分期带有细分字母/数字（如 IIIA1期），你**必须**将其降维简化为大分期（如 "Stage III" OR "Stage IIIA"）！绝对禁止在检索词中保留 "IIIA1"。
        3. **【强制绑定数据库密码】（极度重要！）**：为了获取精确的 5 年/10 年生存率百分比，你**必须**在检索词中强制加入美国国家癌症数据库的检索词组合：(NCDB OR SEER OR "National Cancer Databas")
        4. **拒绝过度拟合**：绝对不要把具体的突变（如 p53, PR-）或具体的合并症塞进预后检索词里！
        5. **【强制年份与后缀】**：必须在末尾加上：AND survival AND 2018:2026[dp]
        
        【正确示范（精简、降维且带有数据库黄金密码）】：
        ("endometrial cancer" OR "endometrial carcinoma") AND "serous" AND ("Stage III" OR "Stage IIIA") AND (NCDB OR SEER OR "National Cancer Database") AND survival AND 2018:2026[dp]
        
        请直接输出最终的英文检索词字符串，不要有任何多余的解释，不要带引号。
        """)
        
        try:
            resp = await invoke_with_timeout_and_retry(self.fast_model, prompt, timeout=120.0)
            
            # 🚀 [修复点 1]：绝对禁止使用 replace('"', '')，必须保留双引号以触发 PubMed 的精确匹配！
            prog_query = remove_think_tags(resp.content).strip()
            # 仅移除可能由于大模型格式化而在最外层包裹的冗余双引号（保证内部双引号不受影响）
            if prog_query.startswith('"') and prog_query.endswith('"') and prog_query.count('"') == 2:
                prog_query = prog_query.strip('"')
                
        except Exception:
            prog_query = '("endometrial cancer" OR "endometrial carcinoma") AND survival'

        logger.info(f"🧬 [Prognosis Track] 专属预后检索词: {prog_query}")
        
        # 2. 独立调用检索工具
        local_selector = ToolSelector(
            self.tool_planning_model, self.reasoning_model, self.mcp_tool_client, available_tools=["search_recent_pubmed"]
        )
        local_executor = ToolExecutor(
            self.mcp_tool_client, self.error_log_path, self.fast_model
        )
        
        try:
            # 这里只需正常下达检索命令
            force_command = f"Search for survival rates using this EXACT query: {prog_query}"
            t_calls = await local_selector.run(force_command)
            
            # 🚀 [修复点 2：终极 Hack 暴力拦截注入] 
            # 无论 LLM 怎么生成工具参数，我们直接在底层 JSON 强行塞入拉取数量上限！
            if t_calls:
                for call in t_calls:
                    if 'tool_input' in call:
                        # 把所有 API 常见的数量参数名全打上，确保绝对万无一失拉取 15 篇！
                        call['tool_input']['max_results'] = 5
                        call['tool_input']['retmax'] = 5
                        call['tool_input']['top_k'] = 5
                        
            # 执行工具调用
            t_results = await local_executor.run(t_calls) or []
        except Exception as e:
            logger.error(f"Prognosis search failed: {e}")
            return "检索预后文献失败。"
            
        # 3. 结果解析与文献库挂载 (🚨 引入极其强壮的脱壳与解析逻辑)
        prog_evidence = ""
        count = 0
        
        for res in t_results:
            # 🚀 [修复点 1：暴力脱壳] 应对各种恶劣的 JSON 嵌套或转义格式
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
                if not block.strip() or "Unknown Title" in block: continue
                
                url = ""
                # 🚀 [修复点 2：强壮的正规表达式] 哪怕只写了 PMID: 1234567 也能完美捕获并生成链接！
                pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', block, re.IGNORECASE) or re.search(r'["\']?(?:PMID|uid|id)["\']?\s*[:=]\s*["\']?(\d{7,9})["\']?', block, re.IGNORECASE)
                
                if pmid_match: 
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
                
                title = "Prognosis Study"
                title_match = re.search(r'^(?:Article )?Title:\s*([^\n]+)', block, re.IGNORECASE | re.MULTILINE)
                if not title_match:
                    title_match = re.search(r'\bTitle:\s*([^\n]+)', block, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                
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
        # =====================================================================
        # 🛡️ [防爆盾机制]：防止 vLLM 400 上下文溢出报错
        # =====================================================================
        if hasattr(self.report_model, 'max_tokens'):
            self.report_model.max_tokens = 4096
        if hasattr(self.report_model, 'max_completion_tokens'):
            self.report_model.max_completion_tokens = 4096
            
        if len(current_knowledge) > 25000:
            logger.warning(f"⚠️ current_knowledge 过长 ({len(current_knowledge)} 字符)，正在执行安全截断...")
            current_knowledge = current_knowledge[:25000] + "\n\n...[前沿证据数据过长，已执行物理截断]..."
        # =====================================================================

        # =====================================================================
        # 🚀 [继承逻辑]：直接从草稿中原样截取“核心指南与共识详尽解析”部分
        # =====================================================================
        original_guideline_text = "## 二、 核心指南与共识详尽解析\n（未能在草稿中匹配到指南解析部分）"
        guideline_match = re.search(r'(## 二、 核心指南与共识详尽解析.*?)(?=\n## 三、|\n## 四、|\Z)', self.treatment_context, re.DOTALL)
        if guideline_match:
            original_guideline_text = guideline_match.group(1).strip()
        # =====================================================================

        # =====================================================================
        # 🤖 智能体 1：核心临床试验“聚类-隔离提取”分治系统 (Map-Reduce)
        # =====================================================================
        async def _run_agent1():
            # 第一步：物理切分文献池
            abstract_dict = {}
            # 匹配 "#### [^^1] 标题 \n 摘要内容" 这种结构
            matches = list(re.finditer(r'(#### \[\^\^(\d+)\](.*?))(?=\n#### \[\^\^\d+\]|\Z)', current_knowledge, re.DOTALL))
            for m in matches:
                ref_id = f"[^^{m.group(2)}]"
                abstract_dict[ref_id] = m.group(1).strip()
                
            if not abstract_dict:
                abstract_dict["[^^ALL]"] = current_knowledge # 正则容错

            # 第二步：调用快速模型进行【模拟聚类】
            cluster_prompt = textwrap.dedent(f"""
            你是一名临床文献分诊专家。以下是我们检索到的多篇文献摘要。
            在医学文献中，同一个重磅试验（如 PORTEC-3、GOG-0258）往往会发表多篇文章。
            为了防止后续分析发生“数据串台（幻觉）”，请你对这些文献进行【聚类归纳】。
            
            【前沿文献库】：
            ---
            {current_knowledge}
            ---
            
            【🚨 核心分诊红线（极度重要）】：
            1. **核心优先**：请优先提取著名的核心临床试验（如 PORTEC, GOG, NRG 系列或大型RCT）。
            2. **最多 4 簇**：你最多只能生成 1 到 4 个高价值的试验簇！
            3. **设立垃圾桶**：对于样本量小、没有明确干预方案（如纯测序综述）、纯基础机制或价值极低的散碎文献，请**必须**统一放入名为 "IGNORE" 的簇中！
            
            【强制输出格式】：
            严格输出一个 JSON，键为“临床试验名称”，值为文献角标列表。
            例如：
            {{
                "PORTEC-3 试验簇": ["[^^1]", "[^^3]"],
                "GOG-0258 试验簇": ["[^^2]"],
                "某靶向联合治疗队列": ["[^^4]"],
                "IGNORE": ["[^^5]", "[^^6]", "[^^7]"]
            }}
            💡 严禁输出除了 JSON 对象外的任何废话！
            """)
            
            cluster_dict = {}
            for attempt in range(2):
                try:
                    cluster_res = await invoke_with_timeout_and_retry(self.fast_model, cluster_prompt, timeout=120.0, max_retries=2)
                    cluster_text = remove_think_tags(cluster_res.content).strip()
                    json_match = re.search(r'\{.*\}', cluster_text, re.DOTALL)
                    if json_match:
                        cluster_dict = json.loads(json_match.group(0))
                        break
                except Exception as e:
                    logger.warning(f"⚠️ 文献聚类 JSON 解析失败 (尝试 {attempt+1}): {e}")
                    
            if not cluster_dict:
                cluster_dict = {"综合试验分析": list(abstract_dict.keys())} # 聚类失败的容错托底

            logger.info(f"🧬 [聚类分诊] 文献已被隔离为 {len(cluster_dict)} 个沙盒: {list(cluster_dict.keys())}")

            # 第三步：沙盒并发定向提取 (Map)
            async def _extract_cluster(trial_name, ref_ids):
                # 🚀 垃圾桶簇直接放弃，节省算力
                if trial_name == "IGNORE" or "IGNORE" in trial_name:
                    return "" 
                    
                focused_knowledge = ""
                for rid in ref_ids:
                    if rid in abstract_dict:
                        focused_knowledge += abstract_dict[rid] + "\n\n"
                    elif rid == "[^^ALL]":
                        focused_knowledge = abstract_dict["[^^ALL]"]
                
                if not focused_knowledge.strip(): return ""
                
                # 常规提取 Prompt
                focused_prompt = textwrap.dedent(f"""
                你是一名顶尖的妇科肿瘤循证医学分析专家。
                这是为你物理隔离出的【{trial_name}】专属文献集。
                
                【当前患者真实病情草稿】：
                {self.treatment_context}
                
                【{trial_name} 专属隔离文献集】：
                ---
                {focused_knowledge}
                ---
                
                【🚨 强制提取红线】：
                1. **【数据真空一票否决】**：如果文献中既没有具体的干预方案，也没有具体的生存获益数据，请直接输出：“已剔除”。
                2. **【防误杀豁免】**：若文献入组标准写“纳入 III 期”，则该患者（IIIA期）完全符合！
                3. **【严格格式】**：
                   #### {trial_name} 是一项 [填写试验类型] [^^x]
                   - **纳入人群**：[罗列标准] 💡 **入组匹配校验**：[明确是否满足]
                   - **分组与干预方案**：[参数详情]
                   - **整体生存获益**：[精确百分比与HR]
                   - **分子分型与亚组分析**：[获益差异]
                   - **毒性与本患者指导意义**：[结合患者合并症]
                
                💡 请先在 <think> 标签内审阅，确认无误后再输出！
                """)
                
                try:
                    # 第一次常规提取
                    res = await invoke_with_timeout_and_retry(self.report_model, focused_prompt, timeout=240.0, max_retries=2)
                    content = remove_think_tags(res.content).strip()
                    
                    # =================================================================
                    # 🛡️ 【Agent 反思与审查机制 (Audit & Rescue)】
                    # 如果判定为“已剔除”，但它明明是个核心簇，强行拦截并打回重做！
                    # =================================================================
                    if "已剔除" in content or "毫无指导价值" in content or content == "":
                        logger.warning(f"⚠️ [审查机制触发] 核心簇【{trial_name}】被异常剔除，启动强制抢救程序！")
                        
                        rescue_prompt = textwrap.dedent(f"""
                        【系统最高级别审计指令】
                        前置提取智能体试图剔除名为【{trial_name}】的文献簇，理由是缺乏具体剂量或生存率数据。
                        现在需要你进行二次审查！
                        
                        【{trial_name} 专属隔离文献集】：
                        {focused_knowledge}
                        
                        ==================================================
                        🚨 【灯塔临床试验导航库（核心白名单宪法）】 🚨
                        你【绝对禁止】凭借你自己的医学常识来判断！你必须【严格、且仅依据】以下名单进行比对匹配：
                        - 早期（I-II期）中低危及中高危：GOG-99, PORTEC-1 或 PORTEC-2
                        - 早期高危（I-II期伴高危因素）及局部晚期（III、IVA期）：PORTEC-3, GOG-0258
                        - 晚期（IVB期）及复发一线：GOG-209, NRG-GY018, RUBY, ATTEND, DUO-E
                        - 晚期复发（二线及以上）：KEYNOTE-775
                        - 分子分型降/升阶梯探索：PORTEC-4a
                        ==================================================
                        
                        【🚨 审查与抢救规则（极度重要）】：
                        1. **【合法淘汰】**：认真核对【{trial_name}】这个名字。如果它**没有明确包含**上述《灯塔临床试验导航库》中的任何一个试验代号（例如：它是某不知名队列、纯基础综述，或者是 GOG-0999 等不在库内的代号），请直接且仅输出：“合法剔除”。
                        2. **【核心抢救（绝对豁免）】**：如果【{trial_name}】明确命中了上述《灯塔临床试验导航库》中的试验名称，则它享有**绝对豁免权**，**绝对不允许剔除**！哪怕摘要里没有具体的毫克剂量和存活率百分比，你也必须抢救它的【定性结论】！
                           请严格按照以下格式强行提取：
                           #### {trial_name} 是一项 [填写试验类型] [^^x]
                           - **纳入人群**：[罗列标准] 
                           - **分组与干预方案**：[简述两组对比策略，如无具体剂量则写“摘要未详述”]
                           - **整体生存获益**：[提取定性结论，例如“显著提高了无复发生存期”或“OS无显著差异”]
                           - **分子分型与亚组分析**：[提取定性结论，例如“p53abn预后最差”]
                           - **毒性与本患者指导意义**：[结合本患者真实合并症给出推论]
                           
                        💡 请务必先在 <think> 标签内执行严格的【词典比对程序】！如果未命中白名单字典，请立刻输出“合法剔除”！
                        """)
                        
                        rescue_res = await invoke_with_timeout_and_retry(self.report_model, rescue_prompt, timeout=240.0, max_retries=1)
                        content = remove_think_tags(rescue_res.content).strip()
                        logger.info(f"✅ [审查机制] 成功抢救核心簇【{trial_name}】数据！")
                    # =================================================================
                        
                    return content
                except Exception as e:
                    logger.warning(f"⚠️ {trial_name} 沙盒提取彻底失败: {e}")
                    return ""

            # 并发执行所有簇的提取
            cluster_tasks = [_extract_cluster(name, ids) for name, ids in cluster_dict.items()]
            cluster_results = await asyncio.gather(*cluster_tasks)
            
            # 第四步：合并与清洗 (Reduce)
            final_analysis = []
            for res in cluster_results:
                if res and "已剔除" not in res and "匹配度极低" not in res and "毫无指导价值" not in res:
                    final_analysis.append(res)
                    
            if not final_analysis:
                return "未发现完全匹配该患者分子分型与分期的核心前瞻性试验数据。"
                
            # 🚀 物理锁：强制最多只保留 4 个高质量试验解析
            return "\n\n".join(final_analysis[:4])

        # =====================================================================
        # 🤖 智能体 1.5：随访方案生成器
        # =====================================================================
        followup_prompt = textwrap.dedent(f"""
        你是一名经验丰富的妇科肿瘤个案管理专家。请结合【患者初步会诊草稿】中患者的真实合并症，撰写一份详尽专业的子宫内膜癌术后随访方案。
        
        【患者初步会诊草稿】：
        {self.treatment_context}

        【强制输出模板】（严禁输出任何Markdown大标题，严格原样复制以下加粗标题并补充）：
        **1. 随访频率**
        - （说明不同时间段的具体复查时间间隔）
        **2. 常规随访内容**
        - **专科查体**：（详细列出重点评估的全身及专科项目）
        - **辅助检查**：（详细列出定期复查的影像与标志物）
        **3. 警示症状**
        - （详细列举提示局部复发或远处转移的具体临床症状）
        **4. 生活方式与合并症管理**
        - （结合草稿中真实的合并症情况，给出针对性的放化疗毒性长程预警与科室随诊建议）
        """)

        async def _run_agent15():
            for attempt in range(2):
                try:
                    res = await invoke_with_timeout_and_retry(self.report_model, followup_prompt, timeout=180.0, max_retries=2)
                    return remove_think_tags(res.content).strip()
                except Exception:
                    pass
            return "随访方案生成失败，请参考指南常规随访。"

        # =====================================================================
        # 🤖 智能体 3：预后数据提取专员
        # =====================================================================
        prognosis_prompt = textwrap.dedent(f"""
        你是一位严谨的肿瘤流行病学数据提取专家。
        任务：阅读以下【预后专属文献库】，提取患者相关的预后生存率数据。

        【当前患者真实病情草稿】：
        {self.treatment_context}
        
        【专属预后文献库】：
        ---
        {prognosis_evidence}
        ---

        【🚨 提取红线】：
        1. **【分期包容性】**：如果文献提供了大类分期（如 Stage III，高危患者总体）的生存率，你必须提取并注明，严禁因为没有精确匹配 IIIA1 子分期而漏杀。
        2. **【零串台幻觉】**：你只能提取与子宫内膜癌相关的预后数据。如果专属库里没找到有效数据或混入了无关疾病，必须直接输出：“当前检索文献未返回有效的特定预后生存数据。”绝对禁止张冠李戴捏造数字！
        3. 必须保留带 ^^ 的角标（如 [^^11]）。
        """)

        async def _run_prognosis_agent():
            for attempt in range(2):
                try:
                    res = await invoke_with_timeout_and_retry(self.report_model, prognosis_prompt, timeout=180.0, max_retries=2)
                    return remove_think_tags(res.content).strip()
                except Exception:
                    pass
            return "暂无具体预后生存率数据。"

        logger.info("🤖 [多智能体并发] 正在执行：试验沙盒聚类隔离解析 & 定制随访 & 预后提取...")
        
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
        1. **【尊重原定权威方案】**：草稿中的主干治疗方案已经过权威校验，你不需要对其药物毒性或禁忌症进行纠错。请直接继承该主要方案。
        2. **【个体化降级绝对优先原则（Toxicity Override）】**：仔细阅读【初步会诊草稿】，如果草稿中因为患者的严重合并症（如心衰、冠心病等）明确提出了**“取消常规放疗”、“降级化疗强度”、“姑息性治疗”**等妥协策略，你**绝对禁止**在最终结论中强行恢复标准的高强度指南方案！你必须坚决贯彻这种“安全优先、带瘤生存”的降级理念，并用它来推翻通用的指南建议。
        3. **【临床试验与PICO解答的无缝融合】**：在最终结论中，结合上方《核心临床试验循证解析》最新数据以及PICO问题，给出循证视角的最终确板意见。
        4. **【预后数据强制量化】**：在【预后分析】中，必须直接引用【预后专员提取的客观事实】中的具体数据和原封不动的角标（如 [^^11]）。
        5. **【合并症的常规转诊】**：将草稿中提到的重大合并症和次要异常，单独列为带有数字序号的项目给出相关科室的转诊建议即可。
        6. **【三占位符机制】**：你绝对不要自己写第二部分、第三部分和第四部分的主体！
           - 在第二部分原封不动输出 `{{{{GUIDELINE_PLACEHOLDER}}}}`
           - 在第三部分原封不动输出 `{{{{TRIAL_PLACEHOLDER}}}}`
           - 在第四部分原封不动输出 `{{{{FOLLOWUP_PLACEHOLDER}}}}`

        ## 📝 必须使用的固定输出模板：

        # 妇科肿瘤 MDT 最终版会诊报告

        ## 一、 病情分析
        ### 1. 病历及病理摘要
        【在此处详尽总结患者病情、高危因素及 FIGO 分期。】
        
        {{{{GUIDELINE_PLACEHOLDER}}}}
        
        ### 3. 核心临床试验及 PICO 循证解析
        {{{{TRIAL_PLACEHOLDER}}}}

        ## 二、 术后处理
        【请直接输出一个连续的数字列表（1、2、3...）】：
        1、 **肿瘤专科最终方案**：【🚨 极其重要：请模仿真实临床医生简洁干练的会诊风格！】
           - 必须包含：**核心方案名称**（如 TC方案 或 单药卡铂）和 **总周期数**（如 6次 或 6周期）。
           - 必须包含：**近期疗效评估规划**（如“建议完成化疗结束3个月复查盆腔增强MRI/上腹部增强CT/两肺平扫CT或PET-CT”）。
           - 💡 **降级表达**：如果是因严重并发症妥协的方案，请自然地在句首带过即可。
           - ❌ **绝对禁止**：不要像药剂师一样罗列“静脉滴注、AUC 5、175mg/m2”等琐碎的处方级给药细节！
           - ✅ **正确示例**：“鉴于患者严重心功能不全，建议取消外照射放疗，行单药卡铂化疗6次。建议完成化疗结束3个月复查盆腔增强MRI/上腹部增强CT及胸部平扫CT。” 
        2、 **分子标志物追踪**：追踪分子分型结果，结合最新 PICO 循证结论指导后续风险管理（如果已明确患者分子分型，这一条可以省略）。
        3、 **【合并症/次要异常 1 名称，如：冠心病】**：建议转诊至【对应科室】门诊进一步评估与随诊，【给出简明建议，如评估放化疗风险】。
        4、 **【合并症/次要异常 2 名称，如：浅表性胃炎】**：建议转诊至消化内科门诊进一步评估与处理。
        【⚠️ 请继续用 5、6、7... 顺延列出患者所有的重大合并症和次要异常，每一个异常必须单独占一条！若全部罗列完毕则结束该段。】

        ## 三、 预后分析
        【结合患者特有高危因素，将上方预后专员提取的数据极其精简干练地串联进来，标明具体数据和文献角标】

        ## 四、 随访方案
        {{{{FOLLOWUP_PLACEHOLDER}}}}
        
        💡 请先在 <think> 标签内思考确认无误后再输出正文模板！
        """)  

        logger.info("🤖 [Agent 2] 正在统筹生成 MDT 报告主干并确保全局逻辑一致性...")
        
        content = ""
        for attempt in range(3):
            try:
                response = await invoke_with_timeout_and_retry(self.report_model, main_prompt, timeout=1200.0, max_retries=3)
                content = remove_think_tags(response.content)
                for cut_word in ["## 五", "# 五", "参考文献", "References"]:
                    if cut_word in content:
                        content = content.split(cut_word)[0].strip()
                break
            except Exception as e:
                logger.error(f"Agent 2 生成报错: {e}")
                if attempt == 2: content = "报告生成失败"

        try:
            # 🚀 替换三占位符
            if "{{GUIDELINE_PLACEHOLDER}}" in content:
                content = content.replace("{{GUIDELINE_PLACEHOLDER}}", original_guideline_text)
            else:
                content = content.replace("### 3. 核心临床试验", f"{original_guideline_text}\n\n### 3. 核心临床试验")

            if "{{TRIAL_PLACEHOLDER}}" in content:
                content = content.replace("{{TRIAL_PLACEHOLDER}}", trial_analysis)
            
            if "{{FOLLOWUP_PLACEHOLDER}}" in content:
                content = content.replace("{{FOLLOWUP_PLACEHOLDER}}", followup_plan)

            new_content, refs_section = self._reindex_references(content)
            full_report = new_content + "\n" + refs_section
            return full_report, full_report 
            
        except Exception as e:
            logger.error(f"Failed to post-process references: {e}")
            fallback_refs = "\n==================================================\n"
            for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                fallback_refs += f"[{i}] URL: {ref.link}\n    Title: {ref.title or ref.link}\n----------\n"
            return current_knowledge + fallback_refs, current_knowledge + fallback_refs
        
        
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
                1. **禁止“以貌取人”**：带有 [专属结构化提纯] 标签的 NCT 或 FDA 数据必须认真评估，若高度相关，请赋予最高优先级！
                2. **严格防噪（宁缺毋滥）**：入组人群与患者明显不符的文献必须果断剔除！
                3. **【灯塔试验绝对豁免权】（最核心红线）**：如果在标题中看到了重磅临床试验的代号（如 PORTEC-3, GOG-0258, RUBY, NRG 等），只要符合该患者领域，你**必须无条件赋予最高优先级选入**！绝对不能把它们当作冗余筛掉！
                4. 坚决剔除垃圾匹配（如普拉提、骨科等无关领域）和基础动物实验。
                
                【强制输出格式】：
                请严格输出一个 JSON 数组，包含你选中的文献编号（最多选5个！）。例如：[0, 2, 4]
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
