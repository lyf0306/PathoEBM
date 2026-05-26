"""Pydantic request/response schemas for the PathoEBM API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class JobSubmitRequest(BaseModel):
    treatment_context: str = Field(
        ...,
        description="Raw Markdown treatment plan text (same format as existing REPL input).",
        min_length=50,
    )
    model_choice: str = Field(
        "auto",
        description="LLM backend: 'auto' (config default), 'local' (vLLM), 'deepseek', or 'gpt'.",
        pattern=r"^(auto|local|deepseek|gpt)$",
    )
    max_iterations: int = Field(
        2,
        ge=1,
        le=5,
        description="Max search iterations (1-5).",
    )
    structured_task: Optional[dict] = Field(
        None,
        description="Optional pre-extracted structured task JSON. Overrides LLM extraction.",
    )


class ProgressSnapshot(BaseModel):
    stage: str = ""
    iteration: int = 0
    message: str = ""
    elapsed_seconds: float = 0.0


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed" | "cancelled"
    created_at: str
    model_choice: str = "auto"
    model_used: str = ""  # actually used model (may differ from choice after fallback)
    progress: ProgressSnapshot | None = None
    error: str | None = None


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    model_used: str = ""
    final_report: str
    elapsed_seconds: float = 0.0
    iterations: int = 0
    reference_count: int = 0


class JobListItem(BaseModel):
    job_id: str
    status: str
    created_at: str
    model_choice: str = "auto"
    model_used: str = ""


class JobListResponse(BaseModel):
    jobs: list[JobListItem]


class HealthResponse(BaseModel):
    status: str = "ok"
    active_jobs: int = 0
    uptime_seconds: float = 0.0


class ErrorDetail(BaseModel):
    detail: str
