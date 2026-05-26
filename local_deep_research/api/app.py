"""FastAPI application for PathoEBM evidence-based medicine pipeline.

Start with:
    cd PathoEBM-main
    uvicorn local_deep_research.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse

# Ensure project root is importable (for absolute imports of local_deep_research)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .models import (
    JobSubmitRequest,
    JobStatusResponse,
    JobResultResponse,
    JobListResponse,
    JobListItem,
    HealthResponse,
    ProgressSnapshot,
)
from .job_manager import JobManager, JobState
from .progress import ProgressReporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model selection (mirrors main.py:run_evidence_update)
# ---------------------------------------------------------------------------

def _create_model(model_choice: str):
    """Create LLM instances with external-first, local-fallback strategy.

    Returns a dict with keys:
        fast_llm:      ChatOpenAI instance for the pipeline
        current_mode:  "deepseek" | "gpt" | "local"
    """
    from ..config import (
        get_local_model,
        get_deepseek_v4,
        get_gpt4_1_mini,
        get_model_provider,
        get_model_fallback,
        check_external_model_health,
        settings,
    )

    if model_choice == "auto":
        model_choice = get_model_provider()

    # ---- Local-only path ----
    if model_choice == "local":
        from ..main import check_local_model_health
        if not check_local_model_health():
            raise RuntimeError("Local vLLM model not available")
        return {
            "fast_llm": get_local_model(temperature=0.1),
            "current_mode": "local",
        }

    # ---- External model path (with fallback) ----
    fallback_mode = get_model_fallback()

    try:
        if model_choice == "deepseek":
            fast_llm = get_deepseek_v4()
            current_mode = "deepseek"
        elif model_choice == "gpt":
            fast_llm = get_gpt4_1_mini()
            current_mode = "gpt"
        else:
            raise ValueError(f"Unknown model provider: {model_choice}")

        if not check_external_model_health(fast_llm):
            raise ConnectionError(f"{model_choice} API unreachable")

        logger.info(f"Using external model: {current_mode}")
        return {"fast_llm": fast_llm, "current_mode": current_mode}

    except Exception as e:
        if fallback_mode == "local":
            logger.warning(
                f"External model '{model_choice}' failed: {e}. "
                f"Falling back to local vLLM."
            )
            from ..main import check_local_model_health
            if not check_local_model_health():
                raise RuntimeError(
                    f"External model '{model_choice}' failed and local model is unavailable"
                )
            return {
                "fast_llm": get_local_model(temperature=0.1),
                "current_mode": "local",
            }
        raise


# ---------------------------------------------------------------------------
# Pipeline execution (mirrors main.py:run_evidence_update)
# ---------------------------------------------------------------------------

async def _execute_job(job: JobState, job_manager: JobManager):
    """Run the full evidence-update pipeline for a single job."""
    try:
        job.start()
        job_manager._write_metadata(job)

        # --- Model selection ---
        models = _create_model(job.model_choice)
        fast_llm = models["fast_llm"]
        current_mode = models["current_mode"]
        job.model_used = current_mode
        job_manager._write_metadata(job)

        # --- Parse input ---
        from ..main import parse_graph_ec_report, extract_structured_task
        from ..config import settings

        report_body, max_index, baseline_refs, separator = parse_graph_ec_report(
            job.treatment_context
        )
        logger.info(f"[Job {job.file_prefix}] Parsed {max_index} baseline references")

        # --- Structured extraction ---
        if job.structured_task_override:
            structured_task = job.structured_task_override
        else:
            structured_task = await extract_structured_task(job.treatment_context, fast_llm)
        structured_task["baseline_references"] = {"max_index": max_index}

        search_payload = {
            "oncology_profile": structured_task.get("oncology_core", {}),
            "critical_infections": structured_task.get("comorbidities", {}).get("critical_infections", []),
            "major_comorbidities_affecting_treatment": structured_task.get("comorbidities", {}).get("major_comorbidities", []),
            "incidental_findings": structured_task.get("comorbidities", {}).get("incidental_findings", []),
            "preliminary_plan": structured_task.get("proposed_plan", {}),
            "specific_pico_questions": structured_task.get("clinical_questions_for_ebm", []),
            "surgery_type": structured_task.get("oncology_core", {}).get("surgery_type", ""),
            "baseline_references": {"max_index": max_index},
        }

        my_target_tools = [
            "search_recent_pubmed",
            "get_studies",
            "get_adverse_reactions_by_drug_name",
            "get_warnings_and_cautions_by_drug_name",
        ]

        # --- Create search system ---
        from ..search_system import AdvancedSearchSystem

        system = AdvancedSearchSystem(
            max_iterations=job.max_iterations,
            questions_per_iteration=settings.detailed.questions_per_iteration,
            is_report=True,
            treatment_context=report_body,
            structured_task=search_payload,
            using_model=current_mode,
            chosen_tools=my_target_tools,
            progress_reporter=job.progress_reporter,
            cancel_event=job.cancel_event,
        )

        # Check cancellation before heavy work
        if job.cancel_event.is_set():
            job.cancel()
            job_manager._write_metadata(job)
            return

        await system.initialize()

        if job.cancel_event.is_set():
            job.cancel()
            job_manager._write_metadata(job)
            return

        query = (
            "Please validate the preliminary treatment plan, "
            "carefully assess the impact of major comorbidities (if any) "
            "on drug toxicity and overall survival, and answer the "
            "specific clinical questions provided."
        )
        results = await system.analyze_topic(query)

        if job.cancel_event.is_set():
            job.cancel()
            job_manager._write_metadata(job)
            return

        # --- Post-process report ---
        from ..utilities.search_utilities import strip_llm_preamble, depersonalize_report

        final_report = results.get("final_report", "")
        if final_report:
            final_report = strip_llm_preamble(final_report)
            final_report = depersonalize_report(final_report)
            final_report = re.sub(r'\*\*', '', final_report)

            # Split references
            split_marker = "=================================================="
            new_evidence_text = final_report
            new_refs_text = ""
            if split_marker in final_report:
                parts = final_report.split(split_marker)
                new_evidence_text = parts[0].strip()
                new_refs_text = parts[1].strip()

            combined_report = (
                "### 循证校验与优化的最终治疗方案 (Deep EBM Synthesized Plan)\n\n"
                f"{new_evidence_text}\n"
                f"\n{separator}\n"
                f"{baseline_refs}\n"
            )
            if new_refs_text:
                combined_report += f"{new_refs_text}\n"

            final_report = combined_report

        # Count references
        ref_count = len(re.findall(r'\[\^\^\d+\]|\[\d+\]', final_report)) if final_report else 0
        iterations = results.get("iterations", job.max_iterations)

        # Save to disk
        job_manager.save_report(job, final_report)
        job.complete(final_report, iterations=iterations, ref_count=ref_count)
        job_manager._write_metadata(job)
        logger.info(f"[Job {job.file_prefix}] Completed in {job.elapsed_seconds:.1f}s")

    except asyncio.CancelledError:
        job.cancel()
        job_manager._write_metadata(job)
        logger.warning(f"[Job {job.file_prefix}] Cancelled")
    except Exception as e:
        job.fail(str(e))
        job_manager._write_metadata(job)
        logger.exception(f"[Job {job.file_prefix}] Failed: {e}")


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create JobManager on startup, graceful shutdown on teardown."""
    job_manager = JobManager(max_concurrent=5)
    app.state.job_manager = job_manager
    cleanup_task = asyncio.create_task(job_manager._purge_loop())
    logger.info("PathoEBM API started (max_concurrent=5, retention=30d)")
    try:
        yield
    finally:
        logger.info("Shutting down...")
        await job_manager.shutdown(grace_period=30.0)
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("PathoEBM API stopped.")


app = FastAPI(
    title="PathoEBM API",
    description="Evidence-based medicine pipeline for gynecologic oncology MDT reports",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Optional API key check
# ---------------------------------------------------------------------------

async def _check_api_key(request: Request):
    """If X-API-Key is configured in secrets, require it on every request."""
    from ..config import get_secret
    expected = get_secret("api", "api_key", default="")
    if not expected:
        return  # No auth configured
    provided = request.headers.get("X-API-Key", "")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    await _check_api_key(request)
    mgr: JobManager = request.app.state.job_manager
    return HealthResponse(
        status="ok",
        active_jobs=mgr.active_count,
        uptime_seconds=mgr.uptime_seconds,
    )


@app.post("/jobs", status_code=201, response_model=JobStatusResponse)
async def submit_job(body: JobSubmitRequest, request: Request):
    await _check_api_key(request)
    mgr: JobManager = request.app.state.job_manager

    try:
        job = mgr.create_job(
            treatment_context=body.treatment_context,
            model_choice=body.model_choice,
            max_iterations=body.max_iterations,
            structured_task_override=body.structured_task,
        )
    except ResourceWarning:
        raise HTTPException(
            status_code=503,
            detail=f"Too many concurrent jobs (max {mgr.max_concurrent}). Try again later.",
        )

    # Launch background execution
    job.task = asyncio.create_task(_execute_job(job, mgr))

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        model_choice=job.model_choice,
        model_used=job.model_used,
    )


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(request: Request):
    await _check_api_key(request)
    mgr: JobManager = request.app.state.job_manager
    jobs = [
        JobListItem(
            job_id=j.job_id,
            status=j.status,
            created_at=j.created_at,
            model_choice=j.model_choice,
            model_used=j.model_used,
        )
        for j in mgr.list_jobs()
    ]
    return JobListResponse(jobs=jobs)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request):
    await _check_api_key(request)
    mgr: JobManager = request.app.state.job_manager
    job = mgr.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = None
    latest = job.progress_reporter.latest
    if latest:
        progress = ProgressSnapshot(
            stage=latest.stage,
            iteration=latest.iteration,
            message=latest.message,
            elapsed_seconds=latest.elapsed_seconds,
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        model_choice=job.model_choice,
        model_used=job.model_used,
        progress=progress,
        error=job.error,
    )


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str, request: Request):
    await _check_api_key(request)
    mgr: JobManager = request.app.state.job_manager
    job = mgr.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "failed":
        raise HTTPException(status_code=409, detail=f"Job failed: {job.error}")

    if job.status in ("pending", "running"):
        raise HTTPException(status_code=404, detail="Job not yet complete")

    report = mgr.load_report(job_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report file not found")

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        model_used=job.model_used,
        final_report=report,
        elapsed_seconds=job.elapsed_seconds,
        iterations=job.iterations,
        reference_count=job.reference_count,
    )


@app.delete("/jobs/{job_id}", response_model=JobStatusResponse)
async def cancel_job(job_id: str, request: Request):
    await _check_api_key(request)
    mgr: JobManager = request.app.state.job_manager
    job = mgr.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if not mgr.cancel_job(job_id):
        raise HTTPException(
            status_code=409,
            detail=f"Job already in terminal state: {job.status}",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        model_choice=job.model_choice,
        model_used=job.model_used,
    )
