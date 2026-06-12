"""
Report generation + reviewer re-generation loop.

Extracted from search_system.py as a mixin for AdvancedSearchSystem.
"""

import asyncio
import logging
import re

from ..agents.treatment_decision_agent import TreatmentDecisionAgent
from ..agents.mdt_report_agent import MDTReportAgent
from ..agents.followup_agent import FollowupAgent
from ..agents.prognosis_agent import PrognosisAgent
from ..agents.reviewer_agent import ReviewerAgent
from ..agents.react_search_agent import ReActSearchAgent
from ..agents.context_bus import AgentContextBus
from ..utilities.search_utilities import remove_think_tags, ensure_chinese_output

logger = logging.getLogger(__name__)


class ReportGenerationMixin:
    """
    Mixin providing _generate_detailed_report + reviewer re-generation loop.

    Expects the host class to provide:
      - self.report_model, self.treatment_context, self.structured_task
      - self.ref_pool, self.progress_reporter, self.fast_model, self.model
      - self._consolidate_trial_analysis, self._filter_irrelevant_trials
      - self._deduplicate_trial_analysis, self._deduplicate_intra_trial
      - self._demote_paper_subheadings, self._fix_numbering_in_postop
      - self._statistical_sanity_check, self._check_citation_credibility
    """

    async def _generate_detailed_report(
        self, current_knowledge: str, findings: list,
        query: str, iteration: int, prognosis_results: dict = None,
        context_bus: AgentContextBus = None,
    ):
        # Allow enough room for R1 thinking + full MDT report output
        if hasattr(self.report_model, 'max_tokens'):
            self.report_model.max_tokens = 16384
        if hasattr(self.report_model, 'max_completion_tokens'):
            self.report_model.max_completion_tokens = 16384

        # ── Save full evidence before any truncation ──
        full_raw_evidence = current_knowledge
        # Expose for fallback reference extraction
        self._full_raw_evidence = full_raw_evidence

        if len(current_knowledge) > 25000:
            logger.warning(f"current_knowledge 过长 ({len(current_knowledge)} 字符)，正在执行安全截断...")
            current_knowledge = current_knowledge[:25000] + "\n\n...[前沿证据数据过长，已执行物理截断]..."

        # Inherit guideline text from draft
        original_guideline_text = "## 二、 核心指南与共识详尽解析\n（未能在草稿中匹配到指南解析部分）"
        guideline_match = re.search(
            r'(## 二、 核心指南与共识详尽解析.*?)(?=\n## 三、|\n## 四、|\Z)',
            self.treatment_context, re.DOTALL
        )
        if guideline_match:
            original_guideline_text = guideline_match.group(1).strip()

        # --- Context Bus: inter-agent communication channel ---
        context_bus = context_bus or AgentContextBus()

        # Extract three prognosis sources
        prognosis_results = prognosis_results or {}
        skill_data = prognosis_results.get("skill_data", "")
        population_data = prognosis_results.get("population", "")
        molecular_data = prognosis_results.get("molecular", "")

        logger.info("多智能体并发: 直接组装ReAct分析 & 定制随访 & 预后提取...")

        # ── Extract trial/PICO analysis directly from ReAct outputs (no ClinicalTrialAgent) ──
        trial_parts = []
        for m in re.finditer(
            r'#### 🎯 (.+?)(?=\n#### 🎯|\n#### 🧬|\n#### 🏥|\Z)',
            full_raw_evidence, re.DOTALL
        ):
            trial_parts.append(m.group(0).strip())

        pico_match = re.search(
            r'#### 🧬 PICO 问题查证(.+?)(?=\n#### 🏥|\Z)',
            full_raw_evidence, re.DOTALL
        )
        if pico_match:
            pico_text = pico_match.group(0).strip()
            # Only include PICO section if it has actual evidence data (not just search direction + LLM instructions)
            instruction_keywords = ['选择标准', '请直接输出', '不要加任何说明', '请输出', '输出你选中']
            pico_evidence_lines = [
                line.strip() for line in pico_text.split('\n')
                if line.strip()
                and '检索方向' not in line
                and not line.strip().startswith('####')
                and not any(kw in line for kw in instruction_keywords)
            ]
            if pico_evidence_lines:
                trial_parts.append(pico_text)
            else:
                logger.info("[PICO] PICO 问题查证无实质内容（仅检索方向或指令残留），已跳过")

        trial_analysis = "\n\n".join(trial_parts) if trial_parts else "（ReAct 检索未产生试验分析数据）"

        # ── Strip confidence tags leaked from ReAct synthesis ──
        trial_analysis = re.sub(
            r'\[[✅⚠️❓🚫]\s*[^\]]*?(?:高置信度|中置信度|低置信度|不可验证)[^\]]*\]',
            '', trial_analysis,
        )

        # ── Extract comorbidity/safety context for MDT agent ──
        safety_match = re.search(
            r'#### 🏥 合并症安全评估(.+?)(?=\Z)',
            full_raw_evidence, re.DOTALL
        )
        comorbidity_context = safety_match.group(0).strip() if safety_match else ""

        # ── Extract HPV / infection / risk factor data from raw evidence ──
        risk_factor_lines = []
        for line in full_raw_evidence.split('\n'):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['hpv', '人乳头瘤', 'tct', '宫颈病', '宫颈癌筛查',
                                                 '阴道断端细胞学', '宫颈上皮内']):
                risk_factor_lines.append(line.strip())
        risk_factor_context = '\n'.join(risk_factor_lines[:30]).strip() if risk_factor_lines else ""
        if risk_factor_context:
            await context_bus.post("ReActAnalysis", "risk_factor_context", risk_factor_context)
            logger.info("[HPV] 从检索证据中提取了 %d 行 HPV/宫颈相关数据", len(risk_factor_lines))

        await context_bus.post("ReActAnalysis", "trial_analysis", trial_analysis)
        if comorbidity_context:
            await context_bus.post("ReActAnalysis", "comorbidity_context", comorbidity_context)

        # --- Run Agent 1.5 (Follow-up), Agent 3 (Prognosis) — no more ClinicalTrialAgent ---
        surgery_type = self.structured_task.get("surgery_type", "")
        followup_agent = FollowupAgent(
            self.report_model, self.treatment_context,
            surgery_type=surgery_type
        )
        oncology_core = self.structured_task.get("oncology_profile", {})
        prognosis_agent = PrognosisAgent(
            self.report_model, self.treatment_context,
            oncology_core=oncology_core,
            context_bus=context_bus,
        )

        try:
            followup_plan, prognosis_data, trial_analysis = await asyncio.wait_for(
                asyncio.gather(
                    followup_agent.run(),
                    prognosis_agent.run(
                        skill_data=skill_data,
                        population_data=population_data,
                        molecular_data=molecular_data,
                    ),
                    self._consolidate_trial_analysis(trial_analysis),
                    return_exceptions=True,
                ),
                timeout=600.0
            )
            if timer := getattr(self, '_timer', None):
                timer.lap("多智能体并发")
            # 处理可能因 return_exceptions=True 产生的异常对象
            if isinstance(followup_plan, BaseException):
                logger.error(f"随访 Agent 异常: {followup_plan}")
                followup_plan = "随访方案生成异常。"
            if isinstance(prognosis_data, BaseException):
                logger.error(f"预后 Agent 异常: {prognosis_data}")
                prognosis_data = "预后数据提取异常。"
            if isinstance(trial_analysis, BaseException):
                logger.error(f"试验分析异常: {trial_analysis}")
                trial_analysis = ""
            await context_bus.post("PrognosisAgent", "prognosis_data",
                                   prognosis_data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"随访/预后/试验合并并发执行超时或崩溃: {e}")
            followup_plan = "随访方案生成超时失败。"
            prognosis_data = "预后数据提取超时失败。"
            await context_bus.post("System", "agent_failure", str(e))

        logger.info(f"预后专员提取结果: {prognosis_data}")

        # Strip all [^^n] citation markers from prognosis data to prevent
        # LLM from citing references that don't match the content.
        prognosis_data = re.sub(r'\[\^\^(\d+)\]', '', prognosis_data)
        prognosis_data = re.sub(r' +', ' ', prognosis_data).strip()

        # --- Patient-trial relevance filter (drop trials that don't match stage/histology) ---
        trial_analysis = self._filter_irrelevant_trials(trial_analysis)

        # --- Title-based deduplication before MDT assembly ---
        trial_analysis = self._deduplicate_trial_analysis(trial_analysis)

        # --- Intra-trial deduplication: detect near-duplicate sub-entries within same trial ---
        trial_analysis = self._deduplicate_intra_trial(trial_analysis)

        # --- Demote paper sub-headings (#### → #####) for visual hierarchy ---
        trial_analysis = self._demote_paper_subheadings(trial_analysis)

        # --- 兜底翻译：确保试验分析在进入下游agent之前已经是中文 ---
        trial_analysis = await ensure_chinese_output(
            trial_analysis, self.fast_model, label="TrialAnalysis", logger=logger
        )

        # ── 去薄：移除仅有标题+一句话、无实质数据点的空壳条目 ──
        trial_analysis = ReActSearchAgent._remove_thin_entries(trial_analysis)

        # --- Agent 2: MDT Report Chief Writer (合并症管理 + Assembly) ---
        mdt_agent = MDTReportAgent(
            self.report_model, self.treatment_context, self.structured_task,
            context_bus=context_bus,
        )

        # --- Agent 2a: Treatment Decision Specialist (主要方案) ---
        treatment_agent = TreatmentDecisionAgent(
            self.report_model, self.structured_task, context_bus=context_bus,
        )

        # Read reference map from ContextBus
        ref_map_str = ""
        if context_bus:
            ref_msgs = await context_bus.get_by_type("reference_map")
            if ref_msgs:
                ref_map_str = ref_msgs[-1]["content"][:3000]

        # Read risk factor context from ContextBus
        risk_factor_context = ""
        if context_bus:
            rf_msgs = await context_bus.get_by_type("risk_factor_context")
            if rf_msgs:
                risk_factor_context = rf_msgs[-1]["content"][:2000]

        guideline_section_demoted = mdt_agent._extract_guideline_section()
        patient_summary = mdt_agent._build_patient_summary()

        # Generate 主要方案 (TreatmentDecisionAgent)
        main_plan = await treatment_agent.generate_main_plan(
            trial_analysis=trial_analysis,
            ref_map_str=ref_map_str,
            patient_summary=patient_summary,
            guideline_section=guideline_section_demoted,
            safety_context=comorbidity_context,
            risk_factor_context=risk_factor_context,
        )
        logger.info(f"[TreatmentDecisionAgent] 主要方案已生成（{len(main_plan)} 字符）")

        content = await mdt_agent.run(
            trial_analysis, followup_plan, prognosis_data,
            main_treatment_plan=main_plan,
            safety_context=comorbidity_context,
        )
        await context_bus.post("MDTReportAgent", "report_generated",
                               f"报告初稿长度: {len(content)} 字符")
        if timer := getattr(self, '_timer', None):
            timer.lap("MDT撰写")

        # --- Reviewer → Agent Re-generation Loop ---
        # ReviewerAgent is a PURE QUALITY CHECKER: it detects issues but does
        # NOT repair them.  Instead, the pipeline feeds issues back to the
        # original generation agents (TreatmentDecisionAgent, MDTReportAgent,
        # FollowupAgent) for targeted re-generation with their full specialized
        # prompts and surgical rules.  Up to 3 review cycles.
        MAX_REVIEW_CYCLES = 3
        RE_REVIEW_TEMP_BUMP = 0.2

        # Extract sub-sections from 术后处理 for potential individual re-generation
        def _extract_main_plan(report: str) -> str:
            m = re.search(r'### 主要方案\n(.*?)(?=\n### 合并症管理|\Z)', report, re.DOTALL)
            return m.group(1).strip() if m else ""

        def _extract_comorbidity_mgmt(report: str) -> str:
            m = re.search(r'### 合并症管理\n(.*?)(?=\n## 三[、.．] 预后分析|\Z)', report, re.DOTALL)
            return m.group(1).strip() if m else ""

        main_plan_content = _extract_main_plan(content)
        comorbidity_content = _extract_comorbidity_mgmt(content)

        original_temp = getattr(self.report_model, 'temperature', None)
        was_repaired = False
        reviewer = ReviewerAgent(self.report_model)

        # Accumulate all issues across review cycles for stateful re-review
        accumulated_issues: dict = {}

        for cycle in range(MAX_REVIEW_CYCLES):
            cycle_label = "初审" if cycle == 0 else f"第{cycle}轮重审"
            logger.info(f"[Reviewer] {cycle_label}开始...")
            if self.progress_reporter:
                self.progress_reporter.report("reviewing", f"审查{cycle_label}中...", iteration=cycle+1)

            # On re-review: pass accumulated previous issues so reviewer can
            # verify each one was actually fixed (stateful "错题本核销" review)
            _prev_for_review = accumulated_issues if accumulated_issues else None
            issues_by_section = await reviewer.review(
                report=content,
                trial_analysis=trial_analysis,
                followup_plan=followup_plan,
                prognosis_data=prognosis_data,
                treatment_context=self.treatment_context,
                context_bus=context_bus,
                surgery_type=surgery_type,
                previous_issues=_prev_for_review,
            )

            if not issues_by_section:
                if cycle == 0:
                    logger.info("[Reviewer] 初审通过，报告质量合格。")
                else:
                    logger.info(f"[Reviewer] {cycle_label}通过——agent 重生成有效，无新问题。")
                break

            # Merge current issues into accumulated set for next cycle's review
            for section, items in issues_by_section.items():
                accumulated_issues.setdefault(section, []).extend(items)
            logger.info(
                f"[Reviewer] 累计问题池: "
                f"{ {s: len(v) for s, v in accumulated_issues.items()} }"
            )

            # ── False-positive / stalemate detection ──
            # If the same section keeps getting flagged across cycles without
            # resolution, treat it as a potential false positive and break.
            if cycle >= 1:
                sections_to_drop = []
                for section, cur_items in issues_by_section.items():
                    acc_count = len(accumulated_issues.get(section, []))
                    cur_count = len(cur_items)
                    if acc_count >= 4 and cur_count > 0:
                        logger.warning(
                            f"[Reviewer] ⚠️ 检测到 [{section}] 章节累计被标记 "
                            f"{acc_count} 次（本轮新增 {cur_count} 个），"
                            f"可能存在误报或争议——跳过该章节的重生成"
                        )
                        sections_to_drop.append(section)
                for section in sections_to_drop:
                    issues_by_section.pop(section, None)

                # If all current issues were detected as stalemate, break
                if not issues_by_section:
                    logger.warning(
                        "[Reviewer] 所有剩余问题被判定为潜在误报/争议，终止审查循环"
                    )
                    break

            total_issues = sum(len(v) for v in issues_by_section.values())
            if total_issues == 0:
                logger.info(f"[Reviewer] {cycle_label}无有效新问题，审查通过。")
                break

            logger.warning(
                f"[Reviewer] {cycle_label}发现 {total_issues} 个问题"
                f"（{list(issues_by_section.keys())}），触发 agent 重生成..."
            )

            # Bump temperature on re-generation for re-review cycles only
            if cycle > 0 and original_temp is not None:
                self.report_model.temperature = original_temp + RE_REVIEW_TEMP_BUMP * cycle
                logger.info(
                    f"[Reviewer] {cycle_label} 重生成 temperature: "
                    f"{original_temp + RE_REVIEW_TEMP_BUMP * (cycle - 1):.1f} → {self.report_model.temperature:.1f}"
                )

            # ── Pre-compute flags & issue lists for all three sections ──
            main_plan_sections = {"主要方案", "病情分析"}
            _needs_main_plan_regen = bool(
                issues_by_section.keys() & main_plan_sections
            )
            _needs_comorbidity_regen = "合并症管理" in issues_by_section
            _needs_followup_regen = "随访方案" in issues_by_section

            main_plan_issues = []
            if _needs_main_plan_regen:
                for sec in main_plan_sections:
                    main_plan_issues.extend(issues_by_section.get(sec, []))
                logger.info(
                    f"[Reviewer] 触发 TreatmentDecisionAgent 重生成 "
                    f"（{len(main_plan_issues)} 个问题）"
                )
                for i, iss in enumerate(main_plan_issues, 1):
                    logger.warning(f"[Reviewer]   → 主要方案 问题#{i}: {iss[:300]}")

            comorbidity_issues = []
            if _needs_comorbidity_regen:
                comorbidity_issues = list(issues_by_section.get("合并症管理", []))
                logger.info(
                    f"[Reviewer] 触发 MDTReportAgent 合并症管理重生成 "
                    f"（{len(comorbidity_issues)} 个问题）"
                )
                for i, iss in enumerate(comorbidity_issues, 1):
                    logger.warning(f"[Reviewer]   → 合并症管理 问题#{i}: {iss[:300]}")

            followup_issues = []
            if _needs_followup_regen:
                followup_issues = list(issues_by_section.get("随访方案", []))
                logger.info(
                    f"[Reviewer] 触发 FollowupAgent 重生成 "
                    f"（{len(followup_issues)} 个问题）"
                )
                for i, iss in enumerate(followup_issues, 1):
                    logger.warning(f"[Reviewer]   → Followup Agent 问题#{i}: {iss[:300]}")

            # ── Execute re-generations (parallelize when safe) ──
            # Dependency: comorbidity needs main_plan (main_treatment_plan=).
            # Followup is independent (no cross-section comorbidity dependency).

            main_plan_task = None
            comorbidity_task = None
            followup_task = None

            # Launch main_plan early if needed (no upstream deps)
            if _needs_main_plan_regen:
                main_plan_task = asyncio.ensure_future(
                    treatment_agent.generate_main_plan(
                        trial_analysis=trial_analysis,
                        ref_map_str=ref_map_str,
                        patient_summary=patient_summary,
                        guideline_section=guideline_section_demoted,
                        safety_context=comorbidity_context,
                        risk_factor_context=risk_factor_context,
                        reviewer_issues=main_plan_issues,
                        previous_output=main_plan_content,
                    )
                )

            # Launch comorbidity early if main_plan NOT being re-gen'd (no dep)
            if _needs_comorbidity_regen and not _needs_main_plan_regen:
                comorbidity_task = asyncio.ensure_future(
                    mdt_agent._generate_comorbidity_management(
                        main_treatment_plan=main_plan_content,
                        safety_context=comorbidity_context,
                        risk_factor_context=risk_factor_context,
                        reviewer_issues=comorbidity_issues,
                        previous_output=comorbidity_content,
                    )
                )

            # Followup is independent — launch early
            if _needs_followup_regen:
                followup_task = asyncio.ensure_future(
                    followup_agent.run(
                        reviewer_issues=followup_issues,
                        previous_output=followup_plan,
                    )
                )

            # ── Resolve main_plan first (comorbidity may depend on it) ──
            if main_plan_task:
                main_plan_content = await main_plan_task

            # Launch comorbidity now if it depends on new main_plan
            if _needs_comorbidity_regen and _needs_main_plan_regen:
                comorbidity_task = asyncio.ensure_future(
                    mdt_agent._generate_comorbidity_management(
                        main_treatment_plan=main_plan_content,
                        safety_context=comorbidity_context,
                        risk_factor_context=risk_factor_context,
                        reviewer_issues=comorbidity_issues,
                        previous_output=comorbidity_content,
                    )
                )

            # ── Resolve comorbidity ──
            if comorbidity_task:
                comorbidity_content = await comorbidity_task

            # ── Resolve followup ──
            if followup_task:
                followup_plan = await followup_task

            # Restore original temperature for next cycle's review
            if original_temp is not None:
                self.report_model.temperature = original_temp

            # ── Re-assemble report with regenerated sections ──
            content = mdt_agent._assemble_final_report(
                mdt_agent._build_patient_summary(),
                mdt_agent._extract_guideline_section(),
                trial_analysis,
                main_plan_content,
                comorbidity_content,
                prognosis_data,
                followup_plan,
            )
            was_repaired = True

        if timer := getattr(self, '_timer', None):
            timer.lap("审查修复")
        if was_repaired:
            await context_bus.post("ReviewerAgent", "repair_applied",
                                   "报告质量问题已通过 agent 重生成修复")
            logger.info("[Reviewer] 报告已通过 agent 重生成修复。")
        else:
            logger.info("[Reviewer] 报告无需修复。")

        # Final safety: strip any lingering think tags from all LLM outputs
        content = remove_think_tags(content)

        # Post-process references
        try:
            # Diagnostic: count [^^n] markers before reindexing
            _n_citations = len(re.findall(r'\[\^\^(\d+)\]', content))
            _n_pool = len(self.ref_pool.pool) if self.ref_pool else 0
            logger.info(
                "[Refs] reindex前: 报告中有 %d 个 [^^n] 引用标记, ref_pool中有 %d 条文献",
                _n_citations, _n_pool,
            )
            new_content, refs_section = self.ref_pool.reindex_references(content)

            # Strip ref_anchor HTML comments (from _trim_visible_refs safety net)
            # These were inserted to ensure all refs appear in the reference list
            # even when visible citations were trimmed to top-3.
            new_content = re.sub(
                r'<!--\s*ref_anchor:\s*(\[\d+\]\s*)+\s*-->',
                '', new_content
            )
            # Collapse blank lines left by stripped anchors
            new_content = re.sub(r'\n{3,}', '\n\n', new_content)

            # Final safety: strip any remaining citation markers from 术后处理 section
            for punct in ['、', '．', '.', '：', ':']:
                pattern = rf"(##\s*二{re.escape(punct)}\s*术后处理[\s\S]*?)(?=\n##\s*[三四]|\Z)"
                m = re.search(pattern, new_content, re.DOTALL)
                if m:
                    section_text = m.group(1)
                    cleaned = re.sub(r'\[\^?\^?\^?\d+(?:\s*[,、，]\s*\^?\^?\^?\d+)*\s*\]', '', section_text)
                    cleaned = re.sub(r' +', ' ', cleaned)
                    cleaned = cleaned.replace(' 。', '。').replace(' ，', '，')
                    new_content = new_content[:m.start()] + cleaned + new_content[m.end():]
                    break

            # Fix numbering disorder in 术后处理 section
            new_content = self._fix_numbering_in_postop(new_content)

            full_report = new_content + "\n" + refs_section

            # Statistical sanity check: intercept X% vs X% with P<0.05
            full_report = self._statistical_sanity_check(full_report)

            return full_report, full_report
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to post-process references: {e}")
            fallback_refs = self._build_fallback_reference_list(
                current_knowledge + "\n" + getattr(self, '_full_raw_evidence', '')
            )
            return current_knowledge + fallback_refs, current_knowledge + fallback_refs

    # =================================================================
    # Fallback reference extraction: scan raw evidence for PMIDs / URLs
    # =================================================================
    @staticmethod
    def _build_fallback_reference_list(raw_text: str) -> str:
        """
        Scan *raw_text* for PubMed PMIDs / URLs and build a minimal
        reference list when the normal ref_pool → reindex_references
        pipeline produced an empty result.

        This is a safety net: it guarantees the report always carries a
        discoverable reference section, even when the structured ref_pool
        is empty or citation markup was lost during synthesis.
        """
        # Extract unique PMIDs in order of first appearance
        seen: set[str] = set()
        refs: list[str] = []

        for m in re.finditer(
            r'(?:pubmed\.ncbi\.nlm\.nih\.gov/|PMID[: ]?\s*)(\d{7,9})',
            raw_text, re.IGNORECASE,
        ):
            pmid = m.group(1)
            if pmid not in seen:
                seen.add(pmid)
                refs.append(
                    f"[{len(refs) + 1}] PMID: {pmid}\n"
                    f"    Title: (请参见原始检索结果)\n"
                    f"    Guidelines: 前沿证据合成 (Deep Research)\n"
                    "----------"
                )

        # Also catch ClinicalTrials.gov NCT IDs
        for m in re.finditer(
            r'(?:clinicaltrials\.gov/study/|NCT)(\d{8})',
            raw_text, re.IGNORECASE,
        ):
            nct = m.group(1)
            nct_full = f"NCT{nct}"
            if nct_full not in seen:
                seen.add(nct_full)
                refs.append(
                    f"[{len(refs) + 1}] {nct_full}\n"
                    f"    Title: (请参见原始检索结果)\n"
                    f"    Guidelines: 前沿证据合成 (Deep Research)\n"
                    "----------"
                )

        if not refs:
            return "\n==================================================\n（无参考文献）\n"

        header = (
            "==================== 参考文献 (References) "
            "====================\n"
        )
        return header + "\n".join(refs) + "\n"

