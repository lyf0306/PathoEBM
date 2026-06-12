import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..search_system_support import safe_json_from_text, extract_and_convert_list
from ..prompts import prompt_manager

logger = logging.getLogger(__name__)


# Known clinical trial name patterns (case-insensitive)
_TRIAL_PATTERN = re.compile(
    r'(PORTEC[-\s]?\d*[a-z]?|GOG[-\s]?\d*[a-z]?|NRG[-\s]?GY\d*|RUBY|'
    r'KEYNOTE[-\s]?\d+)',
    re.IGNORECASE,
)


def _check_multi_trial(query: str) -> str:
    """
    Detect and fix queries that combine different clinical trials with OR.

    Combining distinct trials in one query (e.g. (PORTEC-3 OR GOG-0258)) pollutes
    the LLM context and reduces PubMed precision. When detected, keep only the
    first trial and log a warning.
    """
    # Find all unique trial names in the query
    trial_matches = _TRIAL_PATTERN.findall(query)
    unique_trials = list(dict.fromkeys(t.strip() for t in trial_matches))

    if len(unique_trials) <= 1:
        return query  # Single trial or none — fine

    logger.warning(
        f"检索词包含多个不同试验 ({', '.join(unique_trials)})，"
        f"保留首个试验 {unique_trials[0]}，丢弃其余: {query[:120]}..."
    )

    # Replace the OR-group containing multiple trials with just the first trial
    # Strategy: find the outermost (...) group containing trial names, replace with first trial
    first_trial = unique_trials[0]

    # Try to find a parenthesized group that contains an OR of trial names
    try:
        or_group_match = re.search(
            r'\(\s*' + _TRIAL_PATTERN.pattern + r'(?:\s+OR\s+' + _TRIAL_PATTERN.pattern + r')+\s*\)',
            query, re.IGNORECASE,
        )
    except re.error:
        logger.warning(f"多试验检测正则构造失败，跳过: {query[:100]}...")
        return query
    if or_group_match:
        # Replace the multi-trial OR group with just the first trial name
        fixed = query[:or_group_match.start()] + first_trial + query[or_group_match.end():]
        logger.info(f"多试验检索词修复: {fixed[:150]}...")
        return fixed

    return query  # Can't find OR group — return as-is


def _simplify_query(query: str) -> str:
    """
    Post-generation query validation:
      1. Split multi-trial OR combos (e.g. (PORTEC-3 OR GOG-0258))
      2. Cap AND dimensions at 4 (excluding year filter).
         Raised from 3→4 to accommodate toxicity/safety queries:
         Trial + Disease + Intervention + Toxicity = 4 dimensions.
    """
    # Step 1: Split multi-trial queries
    query = _check_multi_trial(query)

    # Step 2: Check AND-dimension count
    parts = [p.strip() for p in query.split(" AND ")]
    non_year_parts = [p for p in parts if "2018:2026" not in p and "2026" not in p]
    if len(non_year_parts) <= 4:
        return query

    logger.warning(
        f"检索词 AND 条件过多 ({len(non_year_parts)}个)，执行降维: {query[:100]}..."
    )
    year_parts = [p for p in parts if "2018:2026" in p or "2026" in p]

    def _score(p: str) -> int:
        keywords = ["survival", "outcome", "trial", "PORTEC", "GOG", "NRG", "RUBY",
                    "p53", "POLE", "MSI", "mismatch", "chemotherapy", "radiotherapy",
                    "toxicity", "adverse", "safety", "quality of life",
                    "endometrial", "cancer", "carcinoma"]
        return sum(2 for kw in keywords if kw.lower() in p.lower())

    scored = sorted(non_year_parts, key=_score, reverse=True)
    kept = scored[:3]

    simplified = " AND ".join(kept + year_parts)
    logger.info(f"降维结果: {simplified[:150]}...")
    return simplified


def _ensure_date_filter(query: str) -> str:
    """
    安全网：如果检索词没有年份限制，自动追加 2018:2026[dp]。
    仅当 query 看起来是 PubMed 查询（含 AND/OR/[] 等语法）时才追加。
    """
    if "2018:2026" in query or "2026" in query:
        return query
    # 判断是否为 PubMed 检索词（含有 PubMed 语法特征）
    if re.search(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE) or re.search(r'\[.*?\]', query):
        logger.info(f"检索词缺少年份限制，自动追加 AND 2018:2026[dp]: {query[:100]}...")
        return f"({query}) AND 2018:2026[dp]"
    return query


@dataclass
class SearchPlan:
    """
    Structured search plan with per-trial query grouping metadata.

    Fields:
        questions: Flat list of all sub-queries (post-processed).
        trial_plan: Raw parsed trial_search_plan from LLM JSON.
        trial_mapping: List of (trial_name, [indices into questions]).
        pico_indices: Indices in `questions` that belong to PICO.
        comorb_indices: Indices in `questions` that belong to comorbidity.
    """
    questions: List[str] = field(default_factory=list)
    trial_plan: Optional[List[dict]] = None
    trial_mapping: Optional[List[Tuple[str, List[int]]]] = None
    pico_indices: Optional[List[int]] = None
    comorb_indices: Optional[List[int]] = None

    @property
    def has_trial_grouping(self) -> bool:
        return bool(self.trial_mapping)


class SearchPlanner:
    """
    Generates follow-up search questions for iterative evidence retrieval.
    Uses the three-pillar ("三足鼎立") structured strategy to distribute query quotas.
    """

    # ── 灯塔试验导航库 ──
    # 结构化记录每个核心临床试验的适用人群分层（5 级分类）。
    # tier:
    #   "L1_early"              → 早期（I-II期）：中低危及中高危人群 → 做减法
    #   "L2_early_high_la"      → 早期高危（I-II期伴高危因素）及局部晚期（III期） → 做加法
    #   "L3_advanced_1L"         → 晚期（III-IV期）及复发一线治疗 → 化疗+免疫
    #   "L4_advanced_2L"         → 晚期复发（二线及以上治疗） → 靶向+免疫
    #   "L5_molecular_explore"   → 前沿探索：分子分型指导的降阶梯/升阶梯治疗
    # context: 一句话临床定位
    TRIAL_NAVIGATION = {
        # ══════════════════════════════════════════════════════════════
        # L1: 早期（I-II期）：中低危及中高危人群 → 核心是"做减法"
        # ══════════════════════════════════════════════════════════════
        "GOG-99":   {"tier": "L1_early", "context": "早期HIR术后辅助盆腔EBRT（首次定义HIR人群，降低局部复发，无OS获益）"},
        "PORTEC-1": {"tier": "L1_early", "context": "I期中危术后盆腔RT（降低局部复发无OS获益，确立低危患者可仅行观察）"},
        "PORTEC-2": {"tier": "L1_early", "context": "早期HIR → VBT vs EBRT（VBT同等有效+毒性更低+QoL更高=标准治疗，指南级）"},

        # ══════════════════════════════════════════════════════════════
        # L2: 早期高危（I-II期伴高危因素）及局部晚期（III期） → 核心是"做加法"
        # ══════════════════════════════════════════════════════════════
        "GOG-249":  {"tier": "L2_early_high_la",  "context": "早期高危 → VBT+化疗 vs 盆腔EBRT（RFS/OS无差异但化疗毒性更大，不支持常规替代）"},
        "PORTEC-3": {"tier": "L2_early_high_la",  "context": "高危早期+III期 → 辅助放化疗 vs 单纯放疗（III期和浆液性癌显著改善PFS和OS）"},
        "GOG-0258": {"tier": "L2_early_high_la",  "context": "III/IVA期 → 放化疗 vs 单纯化疗（RFS无差异但放化疗局部控制更优，存在争议空间）"},

        # ══════════════════════════════════════════════════════════════
        # L3: 晚期（III-IV期）及复发一线治疗 → 化疗联合免疫治疗
        # ══════════════════════════════════════════════════════════════
        "GOG-209":   {"tier": "L3_advanced_1L", "context": "晚期/复发一线TC方案（紫杉醇+卡铂，非劣效于TAP且毒性更低，确立为标准）"},
        "NRG-GY018": {"tier": "L3_advanced_1L", "context": "晚期/复发一线TC+帕博利珠单抗（dMMR/pMMR均显著延长PFS，dMMR人群获益极其巨大）"},
        "RUBY":      {"tier": "L3_advanced_1L", "context": "晚期/复发一线TC+多塔利单抗（dMMR/MSI-H前所未有PFS和OS获益，pMMR同样改善）"},

        # ══════════════════════════════════════════════════════════════
        # L4: 晚期复发（二线及以上治疗） → 非铂类基础上的靶向与免疫联合
        # ══════════════════════════════════════════════════════════════
        "KEYNOTE-775": {"tier": "L4_advanced_2L", "context": "复发二线+ → 仑伐替尼+帕博利珠单抗 vs 单药化疗（pMMR/MSS二线最重要标杆）"},

        # ══════════════════════════════════════════════════════════════
        # L5: 前沿探索 — 分子分型指导的降阶梯/升阶梯治疗
        # ══════════════════════════════════════════════════════════════
        "PORTEC-4a": {"tier": "L5_molecular_explore", "context": "首个按分子分型（POLE/dMMR/NSMP/p53abn）指导早期HIR辅助治疗的前瞻性随机试验"},
    }

    def __init__(self, tool_planning_model, structured_task: dict, questions_per_iteration: int,
                 treatment_context: str = ""):
        self.tool_planning_model = tool_planning_model
        self.structured_task = structured_task
        self.questions_per_iteration = questions_per_iteration
        self.treatment_context = treatment_context

    @staticmethod
    def _build_trial_navigation_text() -> str:
        """Render TRIAL_NAVIGATION as markdown for the search_planner prompt."""
        lines = [
            "## 灯塔试验导航库（请根据患者病情自主选择匹配的试验）",
            "",
            '### L1 — 早期（I-II期）中低危/中高危 → 核心是“做减法”',
            "",
        ]
        for name in ["GOG-99", "PORTEC-1", "PORTEC-2"]:
            info = SearchPlanner.TRIAL_NAVIGATION[name]
            lines.append(f"- **{name}**：{info['context']}")
        lines.append("")

        lines.append('### L2 — 早期高危（I-II期伴高危因素）及局部晚期（III/IVA期） → 核心是“做加法”')
        lines.append("")
        for name in ["GOG-249", "PORTEC-3", "GOG-0258"]:
            info = SearchPlanner.TRIAL_NAVIGATION[name]
            lines.append(f"- **{name}**：{info['context']}")
        lines.append("")

        lines.append("### L3 — 晚期（III-IV期）及复发一线治疗 → 化疗联合免疫治疗")
        lines.append("")
        for name in ["GOG-209", "NRG-GY018", "RUBY"]:
            info = SearchPlanner.TRIAL_NAVIGATION[name]
            lines.append(f"- **{name}**：{info['context']}")
        lines.append("")

        lines.append("### L4 — 晚期复发（二线及以上治疗） → 靶向联合免疫")
        lines.append("")
        lines.append(f"- **KEYNOTE-775**：{SearchPlanner.TRIAL_NAVIGATION['KEYNOTE-775']['context']}")
        lines.append("")

        lines.append("### L5 — 前沿探索：分子分型指导的降阶梯/升阶梯治疗")
        lines.append("")
        lines.append(f"- **PORTEC-4a**：{SearchPlanner.TRIAL_NAVIGATION['PORTEC-4a']['context']}")
        lines.append("")

        lines.append('**【选择原则】**：')
        lines.append("1. 首先根据患者的 FIGO 分期选择对应的层级（L1-L5）")
        lines.append("2. 再根据风险因素（G3、LVSI+、深肌层浸润、非子宫内膜样组织学）在同层内选择高危/低危匹配的试验")
        lines.append("3. 复发/晚期患者优先选 L3-L4，早期低危优先选 L1")
        lines.append("4. 分子分型明确（NGS确认）的早期HIR患者可额外考虑 L5（PORTEC-4a）")
        lines.append("5. 不确定时宁可多选 1-2 个相关试验，由后续 _filter_irrelevant_trials 兜底过滤")

        return "\n".join(lines)

    @staticmethod
    def _is_molecular_ihc_only(structured_task: dict) -> str:
        """
        检查分子分型是否仅为 IHC 代理指标（未经 NGS 确认）。
        如果是，返回抑制分子特异性检索的提示语；否则返回空字符串。
        """
        core = structured_task.get("oncology_core") or structured_task.get("oncology_profile") or {}
        pathology = (core.get("pathology_and_molecular", "") or "") if isinstance(core, dict) else ""

        # 明确标注结果未出/待确认
        if re.search(r'(结果未出|未回报|待NGS|结果未回报)', pathology):
            return """
        🚨 **【分子分型未明确——禁止分子特异性检索】**：患者的分子分型尚未明确（标注"结果未出/待NGS"），
        **绝对禁止**生成以 p53abn/p53/POLE/MSI/MMRd 等特定分子标志物为核心的检索词。
        所有检索必须基于已确认的临床病理特征（分期、组织学类型、Grade）。"""

        # IHC 代理指标存在但无 NGS 确认
        has_ihc_surrogate = bool(re.search(r'(过表达|免疫组化|IHC)', pathology))
        has_ngs_confirm = bool(re.search(r'(NGS确认|NGS检测|基因测序|高通量测序|二代测序|next.?generation.?sequencing)', pathology, re.IGNORECASE))

        if has_ihc_surrogate and not has_ngs_confirm:
            return """
        🚨 **【分子分型仅 IHC 代理指标——禁止分子特异性检索】**：当前 p53 等信息仅来自免疫组化（IHC）代理指标，
        尚未经 NGS 确认最终 TCGA 分型。根据 WHO/TCGA 多重分类赋予原则，IHC p53abn ≠ 最终分子分型
        （若 NGS 检出 POLE 突变将被重新归类为 POLEmut）。
        **绝对禁止**生成以 p53abn/POLE/MSI 等特定分子标志物为核心的检索词。
        试验检索的"分子亚组"维度应使用通用表述（如 molecular subgroup 或分子分型整体），
        不得使用 p53abn/p53/TP53 等具体标志物名称。"""

        return ""

    async def generate_questions(self, current_knowledge: str, query: str) -> SearchPlan:
        structured_data = json.dumps(self.structured_task, ensure_ascii=False, indent=2)

        # ─── 灯塔试验导航库（LLM 据此自主选择匹配患者病情的试验）───
        nav_text = SearchPlanner._build_trial_navigation_text()

        # ─── 分子分型是否仅为 IHC-only，若是则抑制分子特异性检索 ───
        molecular_restriction = self._is_molecular_ihc_only(self.structured_task)

        prompt = prompt_manager.get("search_planner").format(
            questions_per_iteration=self.questions_per_iteration,
            trial_navigation=nav_text,
            molecular_restriction=molecular_restriction,
            treatment_context=self.treatment_context[:3000],
            structured_data=structured_data,
            current_knowledge=current_knowledge,
        )

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
                    challenge = parsed.get("clinical_challenge", "")
                    if challenge:
                        logger.info(f"翻译官临床质询: {challenge}")

                    # Structured format: trial_search_plan
                    trial_plan_raw = parsed.get("trial_search_plan")
                    if (
                        trial_plan_raw
                        and isinstance(trial_plan_raw, list)
                        and len(trial_plan_raw) > 0
                    ):
                        pico_raw = parsed.get("pico_queries", []) or []
                        comorb_raw = parsed.get("comorbidity_queries", []) or []
                        return self._build_structured_search_plan(
                            trial_plan_raw, pico_raw, comorb_raw,
                        )

                    # Fallback: old flat sub_queries format
                    return self._build_flat_search_plan(parsed)

            # Fallback: JSON parse failed, try list extraction
            fallback_questions = extract_and_convert_list(response_text) or []
            unique_fallback = list(dict.fromkeys(fallback_questions))
            dated_fallback = [_ensure_date_filter(q) for q in unique_fallback[:self.questions_per_iteration]]
            return SearchPlan(questions=dated_fallback)

        except Exception as e:
            logger.warning(f"生成检索问题失败: {e}")
            return SearchPlan(questions=[])

    # ─────────────────────────────────────────────────────────────
    # Structured Search Plan Builders
    # ─────────────────────────────────────────────────────────────

    def _build_structured_search_plan(
        self,
        trial_plan_raw: List[dict],
        pico_queries_raw: List[str],
        comorbidity_queries_raw: List[str],
    ) -> SearchPlan:
        """Build SearchPlan from new trial_search_plan JSON format."""
        flat_questions: List[str] = []
        trial_mapping: List[Tuple[str, List[int]]] = []
        pico_indices: List[int] = []
        comorb_indices: List[int] = []

        # 1. Trial-grouped sub_queries
        for entry in trial_plan_raw:
            trial_name = entry.get("trial", "").strip()
            sub_queries = entry.get("sub_queries", [])
            if not trial_name or not sub_queries:
                continue
            entry_indices: List[int] = []
            for sq in sub_queries:
                if len(flat_questions) >= self.questions_per_iteration:
                    break
                processed = _ensure_date_filter(_simplify_query(sq))
                if processed not in flat_questions:
                    entry_indices.append(len(flat_questions))
                    flat_questions.append(processed)
            if entry_indices:
                trial_mapping.append((trial_name, entry_indices))

        # 2. PICO queries
        for pq in (pico_queries_raw or []):
            if len(flat_questions) >= self.questions_per_iteration:
                break
            processed = _ensure_date_filter(_simplify_query(pq))
            if processed not in flat_questions:
                pico_indices.append(len(flat_questions))
                flat_questions.append(processed)

        # 3. Comorbidity queries
        for cq in (comorbidity_queries_raw or []):
            if len(flat_questions) >= self.questions_per_iteration:
                break
            processed = _ensure_date_filter(_simplify_query(cq))
            if processed not in flat_questions:
                comorb_indices.append(len(flat_questions))
                flat_questions.append(processed)

        n_trial_q = sum(len(indices) for _, indices in trial_mapping)
        logger.info(
            f"结构化检索规划: {len(trial_mapping)} 个试验, "
            f"{n_trial_q} 个试验子检索词, "
            f"{len(pico_indices)} 个PICO, {len(comorb_indices)} 个合并症, "
            f"合计 {len(flat_questions)} 个检索词"
        )

        return SearchPlan(
            questions=flat_questions,
            trial_plan=trial_plan_raw,
            trial_mapping=trial_mapping or None,
            pico_indices=pico_indices or None,
            comorb_indices=comorb_indices or None,
        )

    def _build_flat_search_plan(self, parsed: dict) -> SearchPlan:
        """Build SearchPlan from old flat sub_queries format (no trial grouping)."""
        questions = parsed.get("sub_queries", [])
        if not questions or not isinstance(questions, list):
            return SearchPlan(questions=[])

        unique_questions = list(dict.fromkeys(questions))
        if len(questions) != len(unique_questions):
            logger.info(
                f"检索词优化: 删除了 {len(questions) - len(unique_questions)} 个大模型为了凑数生成的重复词汇。"
            )

        simplified = [_simplify_query(q) for q in unique_questions]
        dated = [_ensure_date_filter(q) for q in simplified]
        return SearchPlan(questions=dated[:self.questions_per_iteration])
