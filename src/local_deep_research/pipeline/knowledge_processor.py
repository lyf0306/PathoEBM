import json
import logging
import textwrap
from typing import Dict, List

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..search_system_support import safe_json_from_text

logger = logging.getLogger(__name__)


class KnowledgeProcessor:
    """
    Handles evidence synthesis, knowledge extraction, and structured data parsing.
    """
    def __init__(self, model, fast_model):
        self.model = model
        self.fast_model = fast_model

    async def answer_query(self, current_knowledge: str, query: str,
                           current_iteration: int, max_iterations: int,
                           existing_refs: list) -> str:
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
            return remove_think_tags(response.content)
        except Exception as e:
            logger.error(f"证据合成失败: {e}")
            return "Error synthesizing evidence."

    async def extract_knowledge(self, facts_md: str, refs_in_round: List[Dict]):
        prompt = f"""
        你是一名顶尖的临床数据挖掘专家，请从以下检索结果中提取具有绝对临床级精度的医学事实。

        【🚨 数据保留致命红线】：
        1. 必须原封不动地保留所有精确数字：药物具体剂量（如 mg/m2, AUC）、放疗参数（如 Gy/次）、周期数。
        2. 必须保留所有统计学硬核数据：具体生存期百分比（OS/DFS/RFS等）、风险比(HR)、95%置信区间(CI)及 P值。
        3. 必须高度关注并保留亚组分析（Subgroup Analysis）和分子分型（如 p53abn, POLE 等）的差异化数据。
        绝对禁止做笼统的语义概括（如"疗效更好"、"显著改善"等废话）！

        Facts:
        {facts_md}

        Refs:
        {json.dumps(refs_in_round)}

        Output JSON:
        {{
            "key_information": "- **[试验名称/主题]**：具体干预方案细节... OS为xx% (HR=xx, 95%CI: xx-xx, P=xx)...",
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

    async def process_multiple_chunks(self, query: str, current_key_info: str) -> str:
        if not current_key_info:
            return current_key_info

        prompt = f"""
        你是一名高级循证医学专家。请将以下医学检索发现整合为一份极其硬核、详细的临床摘要。

        【🚨 致命红线：禁止"洗稿"和概括】：
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

    async def extract_structured_data(self, raw_text: str, source_type: str, query: str) -> str:
        if source_type == "clinicaltrials":
            prompt = textwrap.dedent(f"""
            你是一名顶尖的临床试验数据提取专家。
            以下是来自 ClinicalTrials.gov 的原始 JSON 数据。它包含了大量的冗余字段，请你像"数据榨汁机"一样，提取出最核心的临床信息。

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
        else:
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
            logger.error(f"结构化数据提取失败: {e}")
            return "数据提取失败，未发现有效量化信息。"
