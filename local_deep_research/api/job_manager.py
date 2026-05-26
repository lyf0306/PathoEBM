"""Job lifecycle manager with disk persistence and 30-day auto-cleanup.

Each job produces two files in the output directory:
  {timestamp}_{job_prefix}_report.md
  {timestamp}_{job_prefix}_metadata.json

Storage settings are read from deploy_config.toml [storage] section.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .progress import ProgressReporter

logger = logging.getLogger(__name__)

# ---- Read storage settings from deploy config ----
def _get_storage_config():
    """Read storage settings from deploy_config.toml, with sensible defaults."""
    try:
        from ..config import get_secret
        return {
            "jobs_dir": get_secret("storage", "jobs_dir", "jobs_output"),
            "retention_days": int(get_secret("storage", "retention_days", "30")),
            "cleanup_interval": int(get_secret("storage", "cleanup_interval_seconds", "600")),
        }
    except Exception:
        return {
            "jobs_dir": "jobs_output",
            "retention_days": 30,
            "cleanup_interval": 600,
        }

_storage = _get_storage_config()
OUTPUT_DIR = Path(__file__).resolve().parent / _storage["jobs_dir"]
RETENTION_DAYS = _storage["retention_days"]
CLEANUP_INTERVAL = _storage["cleanup_interval"]


@dataclass
class JobState:
    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed" | "cancelled"
    created_at: str
    model_choice: str
    model_used: str = ""  # actually used model after fallback
    treatment_context: str = ""
    max_iterations: int = 2
    structured_task_override: dict | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    final_report: str = ""
    elapsed_seconds: float = 0.0
    iterations: int = 0
    reference_count: int = 0
    error: str | None = None
    progress_reporter: ProgressReporter = field(default_factory=ProgressReporter)
    _start_time: float = 0.0

    def start(self):
        self.status = "running"
        self._start_time = time.time()

    def complete(self, report: str, iterations: int = 0, ref_count: int = 0):
        self.status = "completed"
        self.final_report = report
        self.elapsed_seconds = round(time.time() - self._start_time, 1)
        self.iterations = iterations
        self.reference_count = ref_count

    def fail(self, error: str):
        self.status = "failed"
        self.error = error
        self.elapsed_seconds = round(time.time() - self._start_time, 1) if self._start_time else 0.0

    def cancel(self):
        self.status = "cancelled"
        self.cancel_event.set()
        self.elapsed_seconds = round(time.time() - self._start_time, 1) if self._start_time else 0.0

    @property
    def file_prefix(self) -> str:
        """Timestamp + first 8 chars of job_id for filename dedup."""
        ts = self.created_at.replace(":", "-").replace("T", "_")[:19]
        short_id = self.job_id[:8]
        return f"{ts}_{short_id}"

    @property
    def report_path(self) -> Path:
        return OUTPUT_DIR / f"{self.file_prefix}_report.md"

    @property
    def metadata_path(self) -> Path:
        return OUTPUT_DIR / f"{self.file_prefix}_metadata.json"


class JobManager:
    """Manages job lifecycle: create, track, cancel, and auto-cleanup."""

    def __init__(self, max_concurrent: int = 5):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobState] = {}
        self.max_concurrent = max_concurrent
        self._start_time: float = time.time()

    # -----------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("pending", "running"))

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 1)

    def create_job(
        self,
        treatment_context: str,
        model_choice: str = "auto",
        max_iterations: int = 2,
        structured_task_override: dict | None = None,
    ) -> JobState:
        if self.active_count >= self.max_concurrent:
            raise ResourceWarning("Too many concurrent jobs")

        job = JobState(
            job_id=uuid.uuid4().hex,
            status="pending",
            created_at=datetime.now().isoformat(),
            model_choice=model_choice,
            treatment_context=treatment_context,
            max_iterations=max_iterations,
            structured_task_override=structured_task_override,
        )
        self._jobs[job.job_id] = job
        self._write_metadata(job)
        logger.info(f"[JobManager] Created job {job.job_id} (prefix={job.file_prefix})")
        return job

    def get_job(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobState]:
        """Return all jobs, newest first."""
        return sorted(
            self._jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("completed", "failed", "cancelled"):
            return False
        job.cancel()
        self._write_metadata(job)
        return True

    def load_report(self, job_id: str) -> str | None:
        job = self._jobs.get(job_id)
        if job is None:
            # Try to find on disk by scanning for metadata files
            for p in OUTPUT_DIR.glob("*_metadata.json"):
                try:
                    meta = json.loads(p.read_text(encoding="utf-8"))
                    if meta.get("job_id") == job_id:
                        report_path = OUTPUT_DIR / f"{Path(p.stem).stem}_report.md"
                        if report_path.exists():
                            return report_path.read_text(encoding="utf-8")
                except Exception:
                    continue
            return None
        if job.final_report:
            return job.final_report
        if job.report_path.exists():
            return job.report_path.read_text(encoding="utf-8")
        return None

    # -----------------------------------------------------------------
    # Disk persistence
    # -----------------------------------------------------------------

    def _write_metadata(self, job: JobState):
        meta = {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "model_choice": job.model_choice,
            "model_used": job.model_used,
            "elapsed_seconds": job.elapsed_seconds,
            "iterations": job.iterations,
            "reference_count": job.reference_count,
            "error": job.error,
        }
        _atomic_write_json(job.metadata_path, meta)

    def save_report(self, job: JobState, report: str):
        _atomic_write_text(job.report_path, report)
        job.final_report = report
        self._write_metadata(job)
        logger.info(f"[JobManager] Report saved: {job.report_path}")

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    async def _purge_loop(self):
        """Background task: periodically delete files older than RETENTION_DAYS."""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)
                self._purge_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[JobManager] Cleanup error")

    def _purge_expired(self):
        cutoff = time.time() - (RETENTION_DAYS * 86400)
        deleted = 0
        for p in OUTPUT_DIR.iterdir():
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    deleted += 1
            except OSError:
                continue
        if deleted:
            logger.info(f"[JobManager] Purged {deleted} expired file(s) (>{RETENTION_DAYS} days)")

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------

    async def shutdown(self, grace_period: float = 30.0):
        """Cancel all running jobs and wait for them to finish."""
        running = [j for j in self._jobs.values() if j.status == "running"]
        if not running:
            return

        logger.info(f"[JobManager] Shutting down, cancelling {len(running)} running job(s)...")
        for job in running:
            job.cancel()

        # Wait for tasks to wind down
        deadline = time.time() + grace_period
        for job in running:
            if job.task and not job.task.done():
                remaining = deadline - time.time()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(job.task, timeout=remaining)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

        for job in running:
            self._write_metadata(job)
        logger.info("[JobManager] Shutdown complete.")


# -----------------------------------------------------------------
# Atomic file writes
# -----------------------------------------------------------------

def _atomic_write_text(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
