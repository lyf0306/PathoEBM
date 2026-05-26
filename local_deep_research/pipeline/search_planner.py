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
    r'(PORTEC[-\s]?\d*[a-z]?|GOG[-\s]?\d*[a-z]?|NRG[-\s]?GY\d*|RUBY|ATTEND|'
    r'DUO[-\s]?E|KEYNOTE[-\s]?\d+)',
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
    def __init__(self, tool_planning_model, structured_task: dict, questions_per_iteration: int,
                 treatment_context: str = ""):
        self.tool_planning_model = tool_planning_model
        self.structured_task = structured_task
        self.questions_per_iteration = questions_per_iteration
        self.treatment_context = treatment_context

    # ─────────────────────────────────────────────────────────────
    # FIGO 驱动灯塔试验解析（纯代码逻辑，不依赖 LLM 判断）
    # ─────────────────────────────────────────────────────────────

    # FIGO 分期 → 风险层级映射。优先 FIGO 2023，其次 2009，最后兜底裸分期。
    # I-II 期 → 早期低危/中危；III-IVA 期 → 局部晚期高危；IVB/复发 → 远处转移
    _FIGO_2023_RE = re.compile(
        r'(IV[A-E]?\d*|III[A-E]?\d*|II[A-E]?\d*|I[A-E]?\d*)'
        r'\s*期?\s*[（(]\s*FIGO\s*2023',
        re.IGNORECASE,
    )
    _FIGO_2009_RE = re.compile(
        r'(IV[A-E]?\d*|III[A-E]?\d*|II[A-E]?\d*|I[A-E]?\d*)'
        r'\s*期?\s*[（(]\s*FIGO\s*2009',
        re.IGNORECASE,
    )

    # 有些诊断文本写了 FIGO 分期但没写年份（如 "FIGO IA2期"），作为兜底
    _FIGO_BARE_RE = re.compile(
        r'(?:FIGO\s+)?(IV[A-E]?\d*|III[A-E]?\d*|II[A-E]?\d*|I[A-E]?\d*)\s*期',
        re.IGNORECASE,
    )

    @staticmethod
    def _parse_figo_stage(stage_str: str) -> Optional[str]:
        """Extract FIGO stage from structured diagnosis text.
        FIGO 2023 takes priority over 2009 when both are present.
        Returns e.g. 'IA2', 'IIIA1', 'IVB'."""
        # 1) FIGO 2023 first (preferred)
        m = SearchPlanner._FIGO_2023_RE.search(stage_str)
        if m:
            return m.group(1).upper()
        # 2) FIGO 2009 fallback
        m = SearchPlanner._FIGO_2009_RE.search(stage_str)
        if m:
            return m.group(1).upper()
        # 3) Bare stage with FIGO prefix: FIGO IA2期, FIGO III期
        m = SearchPlanner._FIGO_BARE_RE.search(stage_str)
        if m:
            return m.group(1).upper()
        return None

    @staticmethod
    def _is_high_risk_histology(text: str) -> bool:
        """Check for non-endometrioid, high-grade histology that confers high risk."""
        t = text.lower()
        return bool(re.search(
            r'(浆液性|serous|透明细胞|clear\s*cell|癌肉瘤|carcinosarcoma|'
            r'非子宫内膜样|non.endometrioid|未分化|undifferentiated)',
            t,
        ))

    @staticmethod
    def _is_high_grade(text: str) -> bool:
        """Check for FIGO Grade 3 (G3 / Ⅲ级).
        Uses re.ASCII so \\b treats CJK chars as non-word boundaries,
        avoiding Unicode \\b quirks without fragile lookbehind syntax."""
        try:
            return bool(re.search(
                r'\b[Gg]3\b|[Gg]rade\s*3|Ⅲ级|III级',
                text, re.ASCII,
            ))
        except re.error as e:
            logger.error(f"[_is_high_grade] 正则异常: {e}")
            return False

    @staticmethod
    def _has_lvsi_positive(text: str) -> bool:
        """True if LVSI is present (not explicitly negated)."""
        try:
            t = text.lower()
            # First check for positive LVSI
            has_lvsi = bool(re.search(
                r'lvsi|脉管癌栓|脉管内癌栓|lymphovascular|vascular\s*invasion',
                t, re.IGNORECASE,
            ))
            if not has_lvsi:
                return False
            # Exclude negated forms
            negated = bool(re.search(
                r'(未见|无|no\s|negative|阴性|denied)[\s\w]{0,20}(lvsi|脉管癌栓|脉管内癌栓|lymphovascular)',
                t,
            ))
            if negated:
                return False
            # Exclude parenthetical negatives: LVSI（-）/ LVSI(-) / LVSI阴性
            if re.search(r'lvsi\s*[（(]\s*[-−—]\s*[）)]|lvsi\s*阴性|lvsi\s*negative', t):
                return False
            return True
        except re.error as e:
            logger.error(f"[_has_lvsi_positive] 正则异常: {e}")
            return False

    @staticmethod
    def _has_deep_myometrial_invasion(text: str) -> bool:
        """True if myometrial invasion >=50% (deep). Excludes superficial/no invasion."""
        try:
            t = text.lower()

            def _negated(pattern: str) -> bool:
                """Check if `pattern` appears in a negated context."""
                return bool(re.search(
                    r'(未见|无|未达|未累及|不累及|排除|否认|no\s|without|absence)'
                    r'[\s\w]{0,30}' + pattern,
                    t,
                ))

            # Negative indicators for deep invasion → return False early
            if re.search(r'浅肌层|浅表肌层|inner\s*half', t):
                return False
            if re.search(r'<50%|<1/2|无肌层|未见肌层', t):
                return False

            # Positive indicators — but only if not negated
            if re.search(r'深肌层|深部肌层', t) and not _negated(r'深肌层|深部肌层'):
                return True
            if re.search(r'deep\s*myometrial|outer\s*half', t) and not _negated(r'deep\s*myometrial|outer\s*half'):
                return True

            # > or ≥ for depth threshold
            if re.search(r'肌层浸润>\s*(?:1/2|50%)', t):
                return True
            if re.search(r'肌层浸润≥\s*(?:1/2|50%)', t):
                return True
            if re.search(r'浸润.*肌层.*>', t):
                return True
            if re.search(r'浸润.*肌层.*≥', t):
                return True

            return False
        except re.error as e:
            logger.error(f"[_has_deep_myometrial_invasion] 正则异常: {e}")
            return False

    @staticmethod
    def _resolve_trial_targets_from_structured(
        figo_stage: Optional[str],
        diag_text: str,
        patho_text: str,
    ) -> str:
        """
        基于从诊断/病理字段提取的结构化变量做决策映射，不再对自由文本做模糊正则。
        """
        combined = f"{diag_text} {patho_text}"

        # 1) 二线及以上治疗
        if re.search(r'(二线|second.line|既往铂类|prior.platinum|铂耐药|platinum.resistant)', combined):
            return "KEYNOTE-775"

        # 2) 复发
        if re.search(r'(复发|recurr|relapse)', combined):
            return "GOG-209、NRG-GY018、RUBY、ATTEND、DUO-E"

        # 3) FIGO 2023 分期判断
        if figo_stage:
            stage = figo_stage.upper()
            # IVB → distant metastasis
            if stage.startswith('IVB'):
                return "GOG-209、NRG-GY018、RUBY、ATTEND、DUO-E"
            # III / IVA → locally advanced → high risk
            if (stage.startswith('III') or stage.startswith('IVA')):
                return "PORTEC-3、GOG-0258"

        # 4) Stage I-II but with high-risk features
        try:
            has_high_risk = (
                SearchPlanner._is_high_risk_histology(combined)
                or SearchPlanner._is_high_grade(combined)
                or SearchPlanner._has_lvsi_positive(combined)
                or SearchPlanner._has_deep_myometrial_invasion(combined)
            )
        except re.error as e:
            logger.error(f"[FIGO Resolver] 高危特征检测正则异常: {e} | diag={diag_text[:100]}")
            has_high_risk = False
        if has_high_risk:
            return "PORTEC-3、GOG-0258"

        # 5) Otherwise: early-stage, low/intermediate risk
        return "GOG-99、PORTEC-1、PORTEC-2"

    def _resolve_trial_targets(self) -> str:
        """从结构化字段提取 FIGO 分期 + 病理特征 → 代码决策映射。"""
        try:
            core = self.structured_task.get("oncology_core") or self.structured_task.get("oncology_profile") or {}
        except Exception:
            core = {}

        diag_text = (core.get("diagnosis_and_stage", "") or "") if isinstance(core, dict) else ""
        patho_text = (core.get("pathology_and_molecular", "") or "") if isinstance(core, dict) else ""

        if not diag_text.strip() and not patho_text.strip():
            diag_text = self.treatment_context[:500]

        # 直接从 diagnosis_and_stage 提取 FIGO 分期（2023 优先，2009 兜底）
        figo_stage = self._parse_figo_stage(diag_text)
        if figo_stage:
            logger.info(f"[FIGO Resolver] 从诊断字段提取 FIGO 分期: {figo_stage}")
        else:
            logger.info("[FIGO Resolver] 未提取到 FIGO 分期，仅依据病理特征判断")

        try:
            targets = self._resolve_trial_targets_from_structured(figo_stage, diag_text, patho_text)
        except re.error as e:
            logger.error(f"[FIGO Resolver] 结构化决策正则异常: {e}，回退至保守策略")
            targets = "GOG-99、PORTEC-1、PORTEC-2"

        # PORTEC-4a 仅适用于早期 HIR（高中危）患者，分子分型指导放疗降/升阶梯。
        # 低危患者（G1、IA、LVSI-）无需辅助治疗，PORTEC-4a 不适用。
        # HIR 特征：深肌层/IB、LVSI+、G3、或 II 期
        try:
            combined = f"{diag_text} {patho_text}"
            is_hir_early = (
                SearchPlanner._has_lvsi_positive(combined)
                or SearchPlanner._has_deep_myometrial_invasion(combined)
                or SearchPlanner._is_high_grade(combined)
                or (figo_stage is not None and figo_stage.upper().startswith('II'))
            )
            has_molecular_data = bool(re.search(
                r'(p53|MSI|MMR|POLE|dMMR|MSH|MLH|PMS2|MSH6|'
                r'分子分型|molecular|NGS|测序|TCGA)',
                patho_text, re.IGNORECASE,
            )) or (
                # IHC/免疫组化仅在有具体结果时才算"有分子数据"，排除"未做/未回报/未出"
                bool(re.search(r'(IHC|免疫组化)', patho_text, re.IGNORECASE))
                and not re.search(r'(IHC|免疫组化).{0,10}(?:未做|未回报|未出|待)', patho_text)
            )
            is_early = bool(re.search(r'(GOG-99|PORTEC-1|PORTEC-2)', targets))
            if has_molecular_data and is_early and is_hir_early and "PORTEC-4a" not in targets:
                targets += "、PORTEC-4a"
                logger.info("[FIGO Resolver] 早期 HIR + 分子数据 → 追加 PORTEC-4a")
        except re.error as e:
            logger.error(f"[FIGO Resolver] PORTEC-4a 检测正则异常: {e}，跳过 PORTEC-4a 判定")

        logger.info(f"[FIGO Resolver] 灯塔试验目标: {targets}")
        return targets

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
        now = datetime.now().strftime("%Y-%m-%d")
        structured_data = json.dumps(self.structured_task, ensure_ascii=False, indent=2)

        # ─────────────────────────────────────────────────────────────
        # 代码级读取 FIGO 分期 → 确定灯塔试验检索目标
        # （不依赖 LLM 判断患者病情，纯逻辑驱动）
        # ─────────────────────────────────────────────────────────────
        trial_targets = self._resolve_trial_targets()

        # ─── 检查分子分型是否为 IHC-only，若是则抑制分子特异性检索 ───
        molecular_restriction = self._is_molecular_ihc_only(self.structured_task)

        prompt = prompt_manager.get("search_planner").format(
            questions_per_iteration=self.questions_per_iteration,
            trial_targets=trial_targets,
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

                    # Try new structured format: trial_search_plan
                    trial_plan_raw = parsed.get("trial_search_plan")
                    if (
                        trial_plan_raw
                        and isinstance(trial_plan_raw, list)
                        and len(trial_plan_raw) > 0
                    ):
                        pico_raw = parsed.get("pico_queries", []) or []
                        comorb_raw = parsed.get("comorbidity_queries", []) or []
                        return self._build_structured_search_plan(
                            trial_plan_raw, pico_raw, comorb_raw
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
