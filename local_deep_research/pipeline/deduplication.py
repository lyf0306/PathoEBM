"""
Deduplication and relevance filtering — trial analysis dedup, non-core filtering.

Extracted from search_system.py as a mixin for AdvancedSearchSystem.
"""

import asyncio
import logging
import re

from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..prompts import prompt_manager

logger = logging.getLogger(__name__)


class DeduplicationMixin:
    """
    Mixin providing deduplication and relevance filtering methods.

    Expects the host class to provide:
      - self.structured_task, self.fast_model
    """

    # =================================================================
    # Non-core relevance pre-filter — code-level disease gate
    # =================================================================
    def _prefilter_non_core_relevance(self, items: list) -> list:
        """
        Quick code-level gate: drop non-core items that are clearly about
        a different disease (wrong cancer type, non-oncologic, etc.).

        This runs BEFORE the LLM _select_non_core_item to save a round-trip
        and to enforce a hard floor on relevance. Only extreme mismatches are
        filtered here; borderline cases are left for the LLM to judge.
        """
        if not items:
            return items

        # Must mention endometrial/uterine cancer or at minimum gynecologic oncology
        disease_pattern = re.compile(
            r'(?i)'
            r'(endometri|uterine|uterus|womb|'
            r'gynecolog|gyn[aec]|cervical|ovarian|vulvar|vaginal|'
            r'PORTEC|GOG|NRG|RUBY|KEYNOTE|'
            r'子宫|内膜|宫颈|卵巢|妇科|外阴|阴道|输卵管|盆腔|附件)'
        )

        # Clearly wrong disease — if the item name-drops another cancer prominently
        # without any mention of endometrial/uterine, drop it
        wrong_disease_pattern = re.compile(
            r'(?i)\b('
            r'lung cancer|non.small.cell.lung|NSCLC|SCLC|'
            r'prostate|breast cancer|colorectal|colon cancer|'
            r'pancreatic|gastric|hepatocellular|HCC|'
            r'glioblastoma|melanoma|leukemia|lymphoma|myeloma|'
            r'head and neck|thyroid|bladder|renal cell|RCC'
            r')\b'
        )

        # Staging methodology papers — no treatment decision value
        staging_methodology_pattern = re.compile(
            r'(?i)('
            r'FIGO\s*(2009|2023|staging).*compar|'
            r'compar.*FIGO\s*(2009|2023|staging)|'
            r'analysing the clinical outcomes between FIGO|'
            r'FIGO.*stage migration|'
            r'staging system.*comparison|'
            r'reclassification.*FIGO|'
            r'FIGO.*reclassif'
            r')'
        )

        # Papers exclusively about a different endometrial histological subtype
        wrong_histology_pattern = re.compile(
            r'(?i)\b('
            r'carcinosarcoma|'
            r'malignant mixed müllerian|'
            r'clear cell carcinoma'
            r')\b'
        )

        kept = []
        for item in items:
            head = item[:300]

            if not disease_pattern.search(head):
                logger.info(
                    "[非核心预过滤] 丢弃: 未提及妇科肿瘤相关疾病 → %s",
                    head.split('\n')[0][:100]
                )
                continue

            endo_mention = bool(re.search(
                r'(?i)(endometri|uterine|uterus|womb)', head
            ))
            wrong_mention = wrong_disease_pattern.search(head)
            if wrong_mention and not endo_mention:
                logger.info(
                    "[非核心预过滤] 丢弃: 研究 %s 与子宫内膜癌无关 → %s",
                    wrong_mention.group(1),
                    head.split('\n')[0][:100]
                )
                continue

            if staging_methodology_pattern.search(head):
                logger.info(
                    "[非核心预过滤] 丢弃: FIGO 分期方法学研究，无治疗决策价值 → %s",
                    head.split('\n')[0][:100]
                )
                continue

            patient_diagnosis = (
                self.structured_task.get("oncology_profile", {})
                .get("diagnosis_and_stage", "")
            )
            if "浆液性" in patient_diagnosis or "serous" in patient_diagnosis.lower():
                wrong_histo = wrong_histology_pattern.search(head)
                if wrong_histo:
                    endo_specific = bool(re.search(
                        r'(?i)(serous|浆液)', head
                    ))
                    if not endo_specific:
                        logger.info(
                            "[非核心预过滤] 丢弃: 非浆液性癌亚型 (%s)，患者为浆液性癌 → %s",
                            wrong_histo.group(1),
                            head.split('\n')[0][:100]
                        )
                        continue

            kept.append(item)

        if len(kept) < len(items):
            logger.info(
                "[非核心预过滤] %d 项 → %d 项 (丢弃 %d 项明显无关的研究)",
                len(items), len(kept), len(items) - len(kept)
            )

        return kept

    # =================================================================
    # LLM-based non-core item selection
    # =================================================================
    async def _select_non_core_item(self, non_core_sections: list) -> list:
        """从非核心条目中用 LLM 选最相关的 1 项，或全部拒绝。"""
        profile = self.structured_task.get("oncology_profile", {}) or {}
        basic_info = profile.get("basic_info", "").strip()
        diagnosis = profile.get("diagnosis_and_stage", "").strip()
        pathology = profile.get("pathology_and_molecular", "").strip()
        patient_context = f"基本信息：{basic_info}\n诊断与分期：{diagnosis}\n病理与分子分型：{pathology}"

        items_text = "\n---\n".join(
            f"【条目 {i+1}】\n{sec}" for i, sec in enumerate(non_core_sections)
        )

        prompt = prompt_manager.get("non_core_selection").format(
            patient_context=patient_context,
            items_text=items_text,
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=300.0, max_retries=3
            )
            selected = remove_think_tags(resp.content).strip()
            if "REJECT_ALL" in selected.upper():
                logger.info(
                    f"非核心条目筛选: {len(non_core_sections)} 项全部与患者无关 → 全部丢弃"
                )
                return []
            logger.info(f"非核心条目筛选完成: {len(non_core_sections)} 项 → 选中 1 项 ({len(selected)} 字符)")
            return [selected]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"非核心条目筛选异常，保留第一条: {e}")
            return [non_core_sections[0]]

    # =================================================================
    # Title-based deduplication — 合并跨节重复论文
    # =================================================================
    @staticmethod
    def _tokenize_title(title: str) -> set:
        """Extract meaningful tokens (>=3 chars) from a title for fuzzy matching."""
        tokens = re.findall(r'[a-z0-9一-鿿]{3,}', title.lower())
        return set(tokens)

    def _deduplicate_trial_analysis(self, trial_analysis: str) -> str:
        """
        Deduplicate paper entries across different trial sections.

        Uses two-stage matching:
          1. Exact match on first 60 chars of normalized title
          2. Token-overlap fuzzy match (>85%) for LLM-reworded duplicates

        The trial_analysis has the structure:
          #### 🎯 PORTEC-3
            #### Molecular Classification of PORTEC-3 [^^8]
            ...
          #### 🎯 GOG-0258
            #### Molecular Classification of PORTEC-3 [^^8]   ← duplicate
            ...

        When the same paper appears under multiple trial sections,
        keep only the first occurrence.
        """
        lines = trial_analysis.split('\n')
        seen_titles: dict = {}  # {clean_title: token_set}
        output = []
        skip = False

        for line in lines:
            stripped = line.strip()

            # Detect any #### header
            if stripped.startswith('#### '):
                # Section headers contain emoji — always reset skip
                if '🎯' in stripped or '🧬' in stripped or '🏥' in stripped:
                    skip = False
                else:
                    # Paper entry header: #### Title [^^n]
                    title_text = re.sub(r'\[\^\^\d+\]', '', stripped).strip()
                    title_text = re.sub(r'^####\s+', '', title_text)
                    clean = re.sub(r'\s+', ' ', title_text.lower())
                    tokens_new = self._tokenize_title(clean)

                    is_dup = False
                    for seen_key, seen_tokens in seen_titles.items():
                        # Stage 1: exact prefix match (first 60 chars)
                        if clean[:60] == seen_key[:60]:
                            is_dup = True
                            break
                        # Stage 2: token-overlap fuzzy match for LLM-reworded titles
                        if len(tokens_new) >= 5 and len(seen_tokens) >= 5:
                            overlap = len(tokens_new & seen_tokens) / min(len(tokens_new), len(seen_tokens))
                            if overlap > 0.85:
                                is_dup = True
                                break

                    if is_dup:
                        skip = True
                        logger.info("[去重] 跳过重复论文: %s...", clean[:60])
                        continue
                    else:
                        seen_titles[clean] = tokens_new
                        skip = False

            if not skip:
                output.append(line)

        return '\n'.join(output)

    # =================================================================
    # Intra-trial deduplication — detect near-duplicate sub-entries within same trial
    # =================================================================
    def _deduplicate_intra_trial(self, trial_analysis: str) -> str:
        """
        Deduplicate sub-entries WITHIN the same trial section.

        The existing _deduplicate_trial_analysis deduplicates papers ACROSS
        different trials (same paper title under PORTEC-3 *and* GOG-0258).
        This catches same-trial duplicates like PORTEC-3 final results [9]
        and PORTEC-3 patterns of recurrence [10] which have nearly identical
        survival data but different titles.

        Strategy: extract numerical fingerprints (%, N=, HR, P values) from each
        sub-entry; if two sub-entries under the same trial share >70% fingerprint
        overlap, keep the content-richer one.
        """
        if not trial_analysis:
            return trial_analysis

        # Split into top-level trial sections
        section_pattern = r'(#### [🎯🧬🏥].+?)(?=\n#### [🎯🧬🏥]|\Z)'
        trial_sections = re.findall(section_pattern, trial_analysis, re.DOTALL)
        if not trial_sections:
            return trial_analysis

        output_sections = []

        for section in trial_sections:
            # Sub-entries are #### lines that DON'T start with the emoji headers
            sub_pattern = r'(#### (?![🎯🧬🏥]).+?)(?=\n#### (?![\n🎯🧬🏥])|\Z)'
            subs = re.findall(sub_pattern, section, re.DOTALL)

            if len(subs) <= 1:
                output_sections.append(section)
                continue

            # Build fingerprints: set of numerical tokens
            fingerprinted = []
            for sub in subs:
                nums = set(re.findall(r'\d+%|N[=:]?\s*\d+|HR\s*[\d.]+|P\s*[=<>]\s*[\d.]+|[\d.]+\s*年', sub))
                fingerprinted.append((sub, nums))

            # Pairwise dedup
            keep = [True] * len(fingerprinted)
            for i in range(len(fingerprinted)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(fingerprinted)):
                    if not keep[j]:
                        continue
                    if not fingerprinted[i][1] or not fingerprinted[j][1]:
                        continue
                    intersection = fingerprinted[i][1] & fingerprinted[j][1]
                    smaller = min(len(fingerprinted[i][1]), len(fingerprinted[j][1]))
                    similarity = len(intersection) / smaller if smaller > 0 else 0
                    if similarity > 0.7:
                        # Keep the longer entry
                        if len(fingerprinted[i][0]) >= len(fingerprinted[j][0]):
                            keep[j] = False
                            logger.info(
                                "[intra-trial 去重] 相似度 %.0f%% — 丢弃 %s",
                                similarity * 100,
                                fingerprinted[j][0][:60].replace('\n', ' ')
                            )
                        else:
                            keep[i] = False
                            logger.info(
                                "[intra-trial 去重] 相似度 %.0f%% — 丢弃 %s",
                                similarity * 100,
                                fingerprinted[i][0][:60].replace('\n', ' ')
                            )
                            break

            # Rebuild section with deduped sub-entries
            kept_subs = [fingerprinted[i][0].strip() for i in range(len(fingerprinted)) if keep[i]]
            # Preserve the section header
            first_line = section.split('\n')[0]
            rebuilt = first_line + '\n\n' + '\n\n'.join(kept_subs)
            output_sections.append(rebuilt)

        result = '\n\n'.join(output_sections)
        if result != trial_analysis:
            logger.info("[intra-trial 去重] 完成: %d 字符 → %d 字符", len(trial_analysis), len(result))
        return result

    @staticmethod
    def _demote_paper_subheadings(trial_analysis: str) -> str:
        """
        Demote #### paper headings to ##### within trial groups.

        After consolidation, the structure is:
          #### 🎯 PORTEC-3        ← trial group header (keep)
          #### Paper Title [^^n]   ← paper entry (demote to #####)
          ...

        This creates a clear visual hierarchy in the final MDT report
        without changing the semantics that downstream agents rely on.
        Done as pure string manipulation — no LLM involved.
        """
        lines = trial_analysis.split('\n')
        result = []
        for line in lines:
            if line.startswith('#### ') and not any(
                emoji in line for emoji in ['🎯', '🧬', '🏥']
            ):
                result.append('#' + line)  # #### → #####
            else:
                result.append(line)
        return '\n'.join(result)

    def _filter_irrelevant_trials(self, trial_analysis: str) -> str:
        """
        Remove core trial sections whose inclusion criteria clearly don't
        match the patient's stage and histology.

        E.g. PORTEC-1 (stage I endometrioid only) is irrelevant for a
        stage III serous patient.
        """
        if not trial_analysis or "超时失败" in trial_analysis:
            return trial_analysis

        # Extract patient characteristics from structured_task
        profile = self.structured_task.get("oncology_profile", {}) or {}
        diagnosis = (profile.get("diagnosis_and_stage", "") or "").lower()
        pathology = (profile.get("pathology_and_molecular", "") or "").lower()

        # Determine patient's stage category
        is_early_stage = bool(re.search(
            r'(?i)[iⅠⅰ][a-c]?\d*\s*期|stage\s*i[abc]?\b|早期',
            diagnosis
        ))
        is_advanced_stage = bool(re.search(
            r'(?i)[iⅠⅰ]{2,}[a-c]?\d*\s*期|stage\s*iii|stage\s*iv|局部晚期|晚期|iiic',
            diagnosis
        ))
        is_endometrioid = bool(re.search(r'(?i)内膜样|endometrioid', pathology))
        has_high_risk = bool(re.search(
            r'(高危|high.risk|g3\b|grade\s*3\b|浆液性|serous|'
            r'lvsi|深肌层浸润|deep.myometrial|non.endometrioid|非子宫内膜样)',
            diagnosis + " " + pathology
        ))
        is_recurrent = bool(re.search(r'(?i)复发|recurr|relapse', diagnosis))
        is_stage4a = bool(re.search(r'(?i)iva\s*期|stage\s*iva\b', diagnosis))
        is_stage4b = bool(re.search(r'(?i)ivb?\s*期|stage\s*iv', diagnosis))
        is_stage4b = is_stage4b and not is_stage4a  # 排除 IVA（归入局部晚期）

        # Trial inclusion criteria (returns True if trial IS relevant to patient)
        trial_rules = {
            # ── Early-stage trials ──
            "PORTEC-1": (    # Stage I endometrioid only
                lambda: is_early_stage and is_endometrioid
            ),
            "PORTEC-2": (    # Stage I-II endometrioid HIR
                lambda: not is_advanced_stage and is_endometrioid
            ),
            "GOG-99": (      # Stage I-II
                lambda: is_early_stage
            ),
            "PORTEC-4a": (   # Stage I HIR — molecular-guided RT (de-escalation/escalation)
                lambda: not is_advanced_stage
            ),
            # ── III / IVA / early-high-risk trials ──
            "PORTEC-3": (    # Stage I high-risk / Stage II / Stage III / IVA
                lambda: is_advanced_stage or has_high_risk
            ),
            "GOG-0258": (    # Stage I high-risk / Stage II / Stage III / IVA
                lambda: is_advanced_stage or has_high_risk or is_recurrent
            ),
            # ── IVB / recurrent-first-line trials ──
            "NRG-GY018": (   # Stage IVB / recurrent (immunotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            "RUBY": (        # Stage IVB / recurrent (immunotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            "GOG-209": (     # Stage IVB / recurrent (chemotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            # ── Second-line ──
            "KEYNOTE-775": ( # Second-line / recurrent
                lambda: is_recurrent
            ),
        }

        # Split by emoji headers ONLY — never split on individual paper titles
        # Primary regex: #### 🎯 / #### 🧬 / #### 🏥
        sections = re.findall(
            r'#### [🎯🧬🏥].+?(?=\n#### [🎯🧬🏥]|\Z)',
            trial_analysis, re.DOTALL
        )

        # Fallback: sections may lack #### prefix after consolidation passes
        if not sections:
            sections = re.findall(
                r'(?:^|\n)(🎯[^\n]*\n.+?)(?=\n🎯|\n🧬|\n🏥|\Z)',
                trial_analysis, re.DOTALL
            )
            if sections:
                logger.warning(
                    "[相关性过滤] 使用降级正则（无####前缀），"
                    "匹配到 %d 个试验段落", len(sections)
                )

        if not sections:
            logger.warning(
                "[相关性过滤] 未匹配到任何试验段落（既无####也无emoji标记），"
                "跳过试验相关性过滤。诊断字段: %s",
                diagnosis[:80] if diagnosis else "(空)"
            )
            return trial_analysis

        kept = []
        removed = []

        for sec in sections:
            first_line = sec.split('\n')[0].strip()
            # Strip heading markers: #### 🎯, 🎯, #### 🧬, etc.
            section_label = re.sub(r'^(?:####\s*)?[🎯🧬🏥]\s*', '', first_line).strip()

            trial_acronym_match = re.search(
                r'\b([A-Z]+-\d+[A-Za-z]*)\b', section_label
            )
            section_trial = trial_acronym_match.group(1) if trial_acronym_match else ''

            should_remove = False
            for trial_name, rule_fn in trial_rules.items():
                if section_trial.lower() == trial_name.lower():
                    if not rule_fn():
                        should_remove = True
                        break

            if should_remove:
                removed.append(section_label)
                logger.info(
                    "[相关性过滤] 移除: %s — 该试验入组人群与患者分期/分型不匹配 "
                    "(诊断: %s, 病理: %s)",
                    section_label[:60], diagnosis[:40], pathology[:40]
                )
            else:
                kept.append(sec)

        if removed:
            logger.info("[相关性过滤] 共移除 %d 项: %s", len(removed), "、".join(r[:30] for r in removed))

        return "\n\n".join(kept) if kept else trial_analysis
