"""Progress reporting for pipeline execution."""

import time
from dataclasses import dataclass, field


@dataclass
class ProgressEvent:
    """A single progress snapshot emitted during pipeline execution."""
    stage: str           # "parsing" | "searching" | "generating" | "reviewing"
    iteration: int = 0   # current iteration (0 for non-iterative stages)
    message: str = ""
    elapsed_seconds: float = 0.0


class ProgressReporter:
    """Collects progress events during a single pipeline run."""

    def __init__(self):
        self.events: list[ProgressEvent] = []
        self._start_time: float = time.time()

    def report(self, stage: str, message: str = "", iteration: int = 0):
        self.events.append(ProgressEvent(
            stage=stage,
            iteration=iteration,
            message=message,
            elapsed_seconds=round(time.time() - self._start_time, 1),
        ))

    @property
    def latest(self) -> ProgressEvent | None:
        return self.events[-1] if self.events else None
