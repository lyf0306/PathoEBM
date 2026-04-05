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
        questions_per_iteration=4,
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
        upstream_report = self.treatment_context
        
        prompt = f"""
        你是一名顶级的“循证医学检索转化专家”（Clinical Evidence Coordinator）。
        上游的 RAG 系统生成了一份【初步 MDT 会诊报告】。

        你的核心任务是：
        1. 【无损提取】：精准理解报告中的“患者全息画像”、“主干治疗方案”及“PICO具体问题”。
        2. 【🚨 临床方案主动质疑（Clinical Challenge）】：主动审视上游推荐的化疗药物（如顺铂）是否属于毒性较大或已过时的传统药物。如果是，必须生成与现代更优替代药物（如卡铂）的“头对头对比”检索词！
        3. 【🚨 破解旧文献陷阱（通用时效性策略！）】：当上游要求查询某经典重磅试验（如 PORTEC、GOG 系列）的“最新数据”时，PubMed 通常会把多年前的初版报告排在第一。为了强制召回【任何年限】的最新随访结果，你绝对不能把年份写死（比如不能只写 10-year，因为有些试验可能是 7年或 5年随访），你**必须使用通用的随访关键词簇，并配合年份标签！**
           - ❌ 错误示范1：PORTEC-3 overall survival (太宽泛，会搜出2018年旧文)
           - ❌ 错误示范2：PORTEC-3 10-year overall survival (太死板，如果该试验只有5年随访数据就会漏检)
           - ✅ 正确示范：PORTEC-3 AND ("follow-up"[Title/Abstract] OR "long-term"[Title/Abstract] OR "updated"[Title/Abstract] OR "final"[Title/Abstract]) AND 2020:2026[dp]
        4. 【转化检索词】：将这些意图转化为能在 PubMed 纯英文数据库中进行精准匹配的高级 Boolean 检索词。

        【上游传入的初步 MDT 会诊报告】：
        {upstream_report}
        
        【当前已验证的知识】：
        {current_knowledge}

        【🚨 检索词设计红线】：
        1. 必须使用纯正的英文医学缩写，绝不能用自然语言长句问问题！
        2. 如果发现了需要质疑的老旧药物，必须生成对比检索词（如：Endometrial cancer Cisplatin Carboplatin efficacy toxicity）。
        3. 必须精确生成 {self.questions_per_iteration} 个英文检索词组。

        【强制输出 JSON 格式】：
        你必须输出合法的 JSON 格式。
        ```json
        {{
            "lossless_extraction": {{
                "patient_status": "极简提炼患者的核心分期、病理、合并症",
                "proposed_treatment": "无损提取上游推荐的主干方案",
                "pico_questions": "提取上游在文末留下的具体待查问题"
            }},
            "clinical_challenge": "记录药物替换质疑，或对抗旧文献的通用策略（如：我使用了 'follow-up' OR 'updated' 组合词，并加上了近期年份标签，以防止漏检）",
            "analysis": "结合全局方案、PICO问题和你的主动质疑，简述检索词构建策略。",
            "sub_queries": [
                "keyword query 1", 
                "keyword query 2",
                "keyword query 3"
            ]
        }}
        ```
        💡 请直接输出 JSON 代码块！
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
        self, current_knowledge: str, findings: List[Dict], query: str, iteration: int
    ):
        # 1. 准备文献标签池 (去掉 ^^，让模型看着最自然)
        pool_text = ""
        for i, r in enumerate(self.ref_pool.pool, 1):
            pool_text += f"[{self.ref_pool.base_idx + i}] {r.title}\n"

        trial_prompt = textwrap.dedent(f"""
        你是一名顶尖的妇科肿瘤（特长：子宫内膜癌）循证医学分析专家。请仔细阅读以下【最新查证的前沿循证数据】，并结合【当前患者真实病情】，为 MDT 会诊报告撰写《核心临床试验循证解析》部分。

        【当前患者真实病情草稿】（🚨 你的核心过滤标准！）：
        {self.treatment_context}
        
        【最新查证的前沿循证数据】（包含详细的原始事实点）：
        ---
        {current_knowledge}
        ---
        文献标签池：
        {pool_text}

        【🚨 核心任务与极度详细的格式红线】：
        1. **【绝对贴合患者特征】**：仔细核对患者的【具体分期】、【病理类型】和【高危因素】。剔除与该患者风险级别不符的文献。
        2. **【拒绝笼统，强制提取数据】**：必须找出具体的生存百分比（如 OS 75%）、HR 值、P 值等硬核量化数据！
        3. **【绝对数量限额】**：你最多只能挑选 1 到 3 个最具决定性临床权重的核心试验！
        4. **【强制输出模板】**：你选出的每个试验必须严格按照以下 4 点结构输出：
        
        #### [填写试验名称或研究主题] [填写对应的文献数字，如 [11] 或 [12] ]
        - **入组人群匹配度**：简述该试验入组标准，并分析其与当前患者特征的匹配情况。
        - **具体干预方案**：明确用药种类、剂量或放疗细节。
        - **关键疗效数据（核心！）**：强制定量！写出具体的 OS、DFS、PFS 百分比、HR 值及 P 值。
        - **毒性与临床意义**：简述关键毒副反应，并给出对本患者的最终指导定论。

        5. **【全中文与自然角标】**：专业缩写保留，其余翻译为中文。直接使用正常的数字角标（如 [11]、[12]），绝对不要自己捏造序号。
        """)
        
        # =====================================================================
        # 🤖 智能体 1.5：子宫内膜癌随访专家 (动态提取合并症版)
        # =====================================================================
        followup_prompt = textwrap.dedent(f"""
        你是一名经验丰富的妇科肿瘤个案管理专家。请根据【患者初步会诊草稿】和【最新查证的前沿循证数据】，撰写一份极其专业的子宫内膜癌术后随访方案。
        
        【患者初步会诊草稿】：
        {self.treatment_context}

        【最新查证的前沿循证数据】：
        {current_knowledge}

        【输出要求】（直接输出正文，严禁输出任何Markdown大标题或总结废话）：
        1. **随访频率**：结合子宫内膜癌指南及前沿数据，明确不同时间段的具体时间间隔。
        2. **常规随访内容**：列出妇科专科查体（如阴道残端检查）及肿瘤标志物（如 CA125、HE4）。
        3. **可能提示复发的警示症状**：列出阴道不规则流血、盆腹腔疼痛等常见复发表现。
        4. **生活方式与毒性管理**：必须高度凝练成一两句话连贯交待，绝对禁止分点展开或列清单！
        """)

        # ---------------------------------------------------------------------
        # ⚡ 异步护栏封装
        # ---------------------------------------------------------------------
        async def _run_agent1_with_guardrail():
            for attempt in range(3):
                try:
                    res = await invoke_with_timeout_and_retry(self.report_model, trial_prompt, timeout=800.0, max_retries=3)
                    content = remove_think_tags(res.content).strip()
                    
                    if len(content) > 3000 or content.count("####") > 4 or "#### 4" in content:
                        logger.warning(f"⚠️ 触发护栏：Agent 1 试验数量超标或异常超长 (尝试 {attempt+1}/3)，打回重做...")
                        if attempt < 2: continue
                    
                    # =====================================================================
                    # 🚀 采用用户的天才思路：兼容模型从1开始编号，以及使用绝对编号的情况
                    # =====================================================================
                    def repl_agent1(match):
                        val = int(match.group(1))
                        # 1. 如果模型乖乖用了绝对编号 [11], [12]：
                        if self.ref_pool.base_idx < val <= self.ref_pool.base_idx + len(self.ref_pool.pool):
                            return f"[^^{val}]"
                        # 2. 如果模型自作主张从 [1], [2] 开始编号：
                        elif 0 < val <= len(self.ref_pool.pool):
                            return f"[^^{self.ref_pool.base_idx + val}]"
                        return match.group(0)
                    
                    content = re.sub(r'\[\s*(\d+)\s*\]', repl_agent1, content)
                    # =====================================================================
                    
                    return content
                except Exception as e:
                    logger.error(f"Agent 1 error: {e}")
                    if attempt == 2: return "临床试验数据解析生成失败，请查阅原始文献。"

        async def _run_agent15():
            try:
                res = await invoke_with_timeout_and_retry(self.report_model, followup_prompt, timeout=800.0, max_retries=3)
                return remove_think_tags(res.content).strip()
            except Exception as e:
                logger.error(f"Agent 1.5 error: {e}")
                return "随访方案生成失败，请参考指南常规随访。"

        logger.info("🤖 [Agent 1 & 1.5] 正在并发执行：生成试验深度解析 & 定制随访规划...")
        trial_analysis, followup_plan = await asyncio.gather(_run_agent1_with_guardrail(), _run_agent15())

        # =====================================================================
        # 🤖 智能体 2：子宫内膜癌 MDT 首席主笔
        # =====================================================================
        main_prompt = textwrap.dedent(f"""
        你是一名具备顶尖国际视野的妇科肿瘤 MDT 首席专家。
        【初步会诊草稿】：
        {self.treatment_context}
        【助手整理的临床试验深度解析】：
        {trial_analysis}

        你的任务是：输出最终版的 MDT 报告主干。
        
        ## 🛑 首席专家“元临床思维”法则 (核心红线)
        1. **【现代临床用药纠偏】**：如果草稿推荐了含铂化疗，请直接在主方案中写明优选卡铂（Carboplatin），以降低毒性。
        2. **【全局用药一致性】**：严密比对患者的合并症，整篇报告的主干方案必须统一口径！
        3. **【预后数据强制量化与自然角标】**：在【预后分析】中，你必须从助手的解析中直接提取具体的生存数据（OS/DFS等）和 HR 值！**并且必须在句尾保留正常的文献角标（如 [11]、[12]），绝对不要漏掉！**
        4. **【双占位符机制】**：在试验解析部分原封不动输出 `{{{{TRIAL_PLACEHOLDER}}}}`；在随访方案部分原封不动输出 `{{{{FOLLOWUP_PLACEHOLDER}}}}`。绝对不要自己写这两个部分！
        5. 绝对不要自己写参考文献列表！绝对不要输出 JSON！写完第四部分立刻停止！

        ## 📝 必须使用的固定输出模板（严格照抄结构，将尖括号 < > 内的说明替换为真实的专业分析，禁止保留尖括号！）：

        # 妇科肿瘤 MDT 最终版会诊报告

        ## 一、 病情分析
        ### 1. 病历及病理摘要
        <在此处详尽总结患者病情、高危因素及 FIGO 分期。并在末尾加上💡临床病理复核建议>

        ### 2. 各大核心指南推荐及风险分层
        - **ESGO指南风险分层**：<填写具体指南的风险分层>
        
        - **ESGO指南推荐方案**：
        `<在此处用代码块包裹原指南推荐的治疗方案。⚠️格式红线：代码块内必须提取并使用高度规格化的医学路径公式（如 "Systemic therapy ± EBRT ± VBT" 或 "Observation"），绝对禁止使用自然语言长句描述！>`
        **分析**：<在此处写出分析。如果有基于患者合并症的用药替换，必须在这里明确指出倾向于使用低毒性的现代替代方案及其理由>
        
        - **其他权威指南推荐方案**（若证据中有）：
        `<在此处用代码块包裹该指南推荐方案，同样必须使用规格化的医学路径公式格式>`
        **分析**：<在此处写出该指南的指导意见分析>
        ### 3. 核心临床试验循证解析
        {{{{TRIAL_PLACEHOLDER}}}}

        ## 二、 术后处理
        ### 1. 肿瘤专科主方案
        <明确写出最终决定的放化疗方案。加上“💡 方案优化与说明”并阐述换药理由>
        
        ### 2. 多学科及合并症管理
        <根据草稿中患者真实的合并症，分点列出各相关科室的随诊建议>

        ## 三、 预后分析
        <必须带有具体的 OS/DFS 百分比数据、HR值！并在句子末尾保留真实的文献角标，如 [11]>

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
                
                # =====================================================================
                # 🛡️ 智能拦截器（Agent 2）：防漏补缺
                # 只有大于指南数量 (base_idx) 的角标才会被判定为新增文献，并打上暗号
                # =====================================================================
                def repl_agent2(match):
                    val = int(match.group(1))
                    if self.ref_pool.base_idx < val <= self.ref_pool.base_idx + len(self.ref_pool.pool):
                        return f"[^^{val}]"
                    return match.group(0)
                    
                content = re.sub(r'\[\s*(\d+)\s*\]', repl_agent2, content)
                # =====================================================================
                
                banned_phrases = ["代码块包裹", "在此处写出", "严禁遗漏", "医学路径公式", "<", ">"]
                lazy_generation_detected = any(phrase in content for phrase in banned_phrases)
                
                if lazy_generation_detected:
                    logger.warning(f"⚠️ 触发护栏：Agent 2 照抄指令或遗留尖括号 (尝试 {attempt + 1}/{max_guardrail_retries})，打回重做...")
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

            # 最终交由排版系统（此时文本中已经藏好了完美的 [^^11] 暗号）
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
            # 💡 【修改点 1】：将其换回 self.model，不再使用额外的小模型
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=1200.0)
            return remove_think_tags(resp.content)
        except Exception:
            return current_key_info

    async def analyze_topic(self, query: str) -> Dict:
        """Main execution loop (回归串行高精度模式，质量永远第一)."""
        logger.info(f"Starting Pure API Validation (Quality-First Serial Mode)")
        
        current_knowledge = ""
        # 🛡️ 依然保留无损证据保险箱，防止最后被宏观总结洗稿
        cumulative_raw_evidence = "" 
        iteration = 0
        findings = []

        await self.initialize()

        while iteration < self.max_iterations:
            questions = await self._get_follow_up_questions(current_knowledge, query)
            if not questions:
                questions = [query]

            # =====================================================================
            # 🛡️ 终极必杀：物理净水器！(绝对不能漏掉这步，否则 API 会一直报 400 错误)
            # =====================================================================
            cleaned_questions = []
            for q in questions:
                # 彻底清除所有的引号、括号等会导致 URL 断裂的特殊符号，以及布尔词和中文
                cq = re.sub(r'[\'\"()[\]{}:;,]', '', q)
                cq = re.sub(r'\b(AND|OR|NOT)\b', '', cq)
                cq = re.sub(r'[\u4e00-\u9fff]', '', cq)
                cq = re.sub(r'\s+', ' ', cq).strip()
                cleaned_questions.append(cq)
            questions = cleaned_questions
            # =====================================================================
                
            self.questions_by_iteration[iteration] = questions
            
            logger.info(f"🚀 Iteration {iteration+1}: Serially processing {len(questions)} sub-questions for maximum reasoning quality...")
            
            fullquery_tool_results = []
            
            # =====================================================================
            # 🔙 核心恢复：回归原始的串行循环！
            # =====================================================================
            for question in questions:
                logger.info(f"🔍 Deep processing question: {question}")
                try:
                    tool_calls = await self.tool_selector.run(question)
                except Exception as e:
                    logger.warning(f"Tool selection failed: {e}")
                    tool_calls = []

                if not tool_calls:
                    continue

                try:
                    tool_results = await self.tool_executor.run(tool_calls) or []
                except Exception as e:
                    logger.error(f"Tool execution failed for question '{question}': {e}")
                    tool_results = []
                
                if tool_results:
                    # 🛡️ 纯代码防御：提取安全文献URL，以防模型在压缩时搞丢角标
                    safe_refs = []
                    for res in tool_results:
                        res_str = str(res)
                        for pmid in set(re.findall(r'PMID:\s*(\d+)', res_str, re.IGNORECASE)):
                            safe_refs.append({"url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "description": f"PMID: {pmid}"})
                        for nct in set(re.findall(r'NCT\d{8}', res_str, re.IGNORECASE)):
                            safe_refs.append({"url": f"https://clinicaltrials.gov/study/{nct}", "description": f"Trial: {nct}"})

                    try:
                        # 恢复原版的 parse_single 单发提纯
                        parsed_list = await asyncio.gather(
                            *(parse_single(res, query=question) for res in tool_results),
                            return_exceptions=True
                        )
                        valid_parsed = [p for p in parsed_list if not isinstance(p, Exception)]
                        
                        if valid_parsed:
                            # 恢复原版的 compress_all_llm，精准提炼 Top 3 事实！
                            compressed = await compress_all_llm(
                                self.model, valid_parsed, limit=3, query=query
                            )
                            # 强制绑定安全 URL
                            if isinstance(compressed, list):
                                for c in compressed:
                                    if isinstance(c, dict):
                                        c["references"] = c.get("references", []) + safe_refs
                            else:
                                compressed = [{"extracted_facts": [str(compressed)], "references": safe_refs}]
                                
                            fullquery_tool_results.extend(compressed)
                    except Exception as e:
                        logger.warning(f"Error parsing tool results: {e}")
            # =====================================================================

            iteration += 1
            facts, refs_raw = [], []
            for item in fullquery_tool_results:
                facts.extend(item.get("extracted_facts", []))
                refs_raw.extend(item.get("references", []))

            unique_refs = {}
            for ref in refs_raw:
                url = ref.get("url", "").strip()
                if url and url not in unique_refs:
                    unique_refs[url] = ref
            refs = list(unique_refs.values())
            self.all_links_of_system.extend([r["url"] for r in refs])

            facts_md = "\n".join(f"- {f}" for f in facts)
            
            # 使用 _extract_knowledge 进一步格式化
            key_info, cleaned_refs = await self._extract_knowledge(facts_md, refs)
            if not cleaned_refs:
                cleaned_refs = refs

            self.knowledge_chunks.append({"key_info": key_info})
            
            # =========================================================================
            # 🛡️ 核心修复 1：必须存入未经 JSON 压缩的原汁原味数据 facts_md！
            # =========================================================================
            chunk_evidence = f"\n\n### 第 {iteration} 轮检索到的详细医学事实：\n{facts_md}\n\n💡 本轮对应可引用的文献角标如下：\n"
            for ref in cleaned_refs:
                ref_url = ref.get("url", "")
                if not ref_url: continue
                ref_title = ref.get("description") or ref.get("title") or ref_url
                idx = self.ref_pool.add(title=ref_title, citation="", link=ref_url)
                chunk_evidence += f"- 文献来源: {ref_title} -> 可引用的真实角标: [^^{idx}]\n"

            cumulative_raw_evidence += chunk_evidence

            current_knowledge = await self.process_multiple_knowledge_chunks(query, key_info)
            final_answer = await self._answer_query(
                current_knowledge, query, iteration, self.max_iterations
            )
            # 原版的粗略总结用于后续轮次的宏观指引
            current_knowledge = final_answer

        final_report = ""
        if self.is_report:
            try:
                # 🎯 把最详尽的保险箱数据交给 Agent 1 进行最终排版
                final_report_tuple = await self._generate_detailed_report(
                    cumulative_raw_evidence, findings, query, iteration
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
