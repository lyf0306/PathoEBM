#!/usr/bin/env python3
"""
PathoEBM Concurrency Stress Test
=================================
Tests the concurrency infrastructure under various load scenarios to find
throughput limits, bottleneck saturation points, and degradation patterns.

Modes:
  --mode primitives  : Stress-test concurrency primitives in isolation (zero deps)
  --mode simulate    : Simulate full pipeline with mock LLM/MCP delays
  --mode live        : Run against real endpoints (costs API credits!)

Usage:
  python scripts/stress_test.py --mode primitives
  python scripts/stress_test.py --mode simulate --workers 1 5 10 15 20 30
  python scripts/stress_test.py --mode live --test-file /root/PathoEBM/local_deep_research/test_3.md
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stress_test")

# ══════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════


@dataclass
class TaskMetrics:
    task_id: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    success: bool = False
    error: str = ""


@dataclass
class RunSummary:
    label: str
    config: dict
    total_tasks: int
    completed: int
    failed: int
    cancelled: int
    durations: list[float] = field(default_factory=list)
    throughput_per_min: float = 0.0
    wall_clock: float = 0.0
    peak_memory_mb: float = 0.0
    semaphore_wait_times: list[float] = field(default_factory=list)
    queue_depths: list[int] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return _percentile(self.durations, 50)

    @property
    def p95(self) -> float:
        return _percentile(self.durations, 95)

    @property
    def p99(self) -> float:
        return _percentile(self.durations, 99)

    @property
    def mean(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.durations) if len(self.durations) > 1 else 0.0

    @property
    def success_rate(self) -> float:
        return self.completed / self.total_tasks if self.total_tasks > 0 else 0.0


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# ══════════════════════════════════════════════════════════════════
# Inline concurrency primitives (zero external deps)
# ══════════════════════════════════════════════════════════════════


class _SimpleTaskQueue:
    """Minimal asyncio.Queue wrapper matching AsyncioTaskQueue interface."""

    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._completed = 0
        self._enqueued = 0

    async def put(self, item):
        self._enqueued += 1
        await self._queue.put(item)

    async def get(self):
        return await self._queue.get()

    def task_done(self):
        self._completed += 1
        self._queue.task_done()

    async def join(self):
        await self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def total_completed(self) -> int:
        return self._completed

    @property
    def total_enqueued(self) -> int:
        return self._enqueued


class _TokenBucketLimiter:
    """Async-safe token bucket rate limiter."""

    def __init__(self, name: str, rate: float, capacity: float):
        self.name = name
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                self._tokens = 0.0
                await asyncio.sleep(wait)
                # Refill after sleep
                now2 = time.monotonic()
                elapsed2 = now2 - self._last_refill
                self._tokens = min(self.capacity, self._tokens + elapsed2 * self.rate)
                self._last_refill = now2
            self._tokens -= 1.0


class _CircuitBreaker:
    """Three-state circuit breaker (CLOSED / OPEN / HALF_OPEN)."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0, window_seconds: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.window_seconds = window_seconds
        self.state = self.CLOSED
        self._failures: list[float] = []
        self._lock = asyncio.Lock()
        self._opened_at: float = 0.0

    async def call(self, coro_factory, *args, **kwargs):
        async with self._lock:
            now = time.monotonic()
            # Purge old failures
            self._failures = [t for t in self._failures if now - t < self.window_seconds]
            if self.state == self.OPEN:
                if now - self._opened_at >= self.recovery_timeout:
                    self.state = self.HALF_OPEN
                else:
                    raise _CircuitBreakerOpenError(self.name)
        try:
            if inspect.iscoroutinefunction(coro_factory):
                result = await coro_factory(*args, **kwargs)
            else:
                result = coro_factory(*args, **kwargs)
            async with self._lock:
                if self.state == self.HALF_OPEN:
                    self.state = self.CLOSED
                    self._failures.clear()
            return result
        except Exception:
            async with self._lock:
                self._failures.append(time.monotonic())
                if len(self._failures) >= self.failure_threshold:
                    self.state = self.OPEN
                    self._opened_at = time.monotonic()
            raise


class _CircuitBreakerOpenError(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Circuit breaker [{name}] is OPEN")


# ══════════════════════════════════════════════════════════════════
# Part 1 — Concurrency Primitive Stress Tests
# ══════════════════════════════════════════════════════════════════


async def _test_task_queue(num_tasks: int, num_workers: int,
                           task_delay: tuple[float, float]) -> RunSummary:
    """Stress-test task queue with configurable workers and task delays."""
    queue = _SimpleTaskQueue()
    done = asyncio.Event()
    metrics: list[TaskMetrics] = []
    queue_depths: list[int] = []
    t0 = time.perf_counter()

    for i in range(num_tasks):
        delay = random.uniform(*task_delay)
        await queue.put((i, delay))

    async def _worker():
        while not done.is_set():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if done.is_set() and queue.empty():
                    break
                continue
            idx, delay = item
            tm = TaskMetrics(task_id=str(idx), start_time=time.perf_counter())
            try:
                await asyncio.sleep(delay)
                tm.end_time = time.perf_counter()
                tm.duration = tm.end_time - tm.start_time
                tm.success = True
            except asyncio.CancelledError:
                tm.error = "cancelled"
            except Exception as e:
                tm.error = str(e)
            metrics.append(tm)
            queue_depths.append(queue.qsize())
            queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(num_workers)]
    await queue.join()
    done.set()
    await asyncio.gather(*workers, return_exceptions=True)

    durations = [m.duration for m in metrics if m.success]
    return RunSummary(
        label=f"Queue({num_tasks}t_{num_workers}w)",
        config={"num_tasks": num_tasks, "num_workers": num_workers, "task_delay": task_delay},
        total_tasks=num_tasks,
        completed=sum(1 for m in metrics if m.success),
        failed=sum(1 for m in metrics if m.error and m.error != "cancelled"),
        cancelled=sum(1 for m in metrics if m.error == "cancelled"),
        durations=durations,
        throughput_per_min=len(durations) / ((time.perf_counter() - t0) / 60) if durations else 0.0,
        wall_clock=time.perf_counter() - t0,
        queue_depths=queue_depths,
    )


async def _test_semaphore_contention(num_tasks: int, semaphore_limit: int,
                                     task_delay: float) -> RunSummary:
    """Stress-test asyncio.Semaphore contention with instrumented wait times."""
    sem = asyncio.Semaphore(semaphore_limit)
    metrics: list[TaskMetrics] = []
    wait_times: list[float] = []
    t0 = time.perf_counter()

    async def _worker(idx: int):
        tm = TaskMetrics(task_id=str(idx), start_time=time.perf_counter())
        wait_start = time.perf_counter()
        async with sem:
            wait_times.append(time.perf_counter() - wait_start)
            try:
                delay = task_delay * (1.0 if random.random() > 0.2 else 3.0)
                await asyncio.sleep(delay)
                tm.end_time = time.perf_counter()
                tm.duration = tm.end_time - tm.start_time
                tm.success = True
            except Exception as e:
                tm.error = str(e)
        metrics.append(tm)

    tasks = [asyncio.create_task(_worker(i)) for i in range(num_tasks)]
    await asyncio.gather(*tasks, return_exceptions=True)

    durations = [m.duration for m in metrics if m.success]
    return RunSummary(
        label=f"Semaphore({num_tasks}t_lim{semaphore_limit})",
        config={"num_tasks": num_tasks, "semaphore_limit": semaphore_limit, "task_delay": task_delay},
        total_tasks=num_tasks,
        completed=sum(1 for m in metrics if m.success),
        failed=sum(1 for m in metrics if m.error),
        cancelled=0,
        durations=durations,
        throughput_per_min=len(durations) / ((time.perf_counter() - t0) / 60) if durations else 0.0,
        wall_clock=time.perf_counter() - t0,
        semaphore_wait_times=wait_times,
    )


async def _test_bounded_gather(num_coros: int, concurrency: int,
                               task_delay: float) -> RunSummary:
    """Stress-test bounded (semaphore-gated) gather with varying concurrency limits."""
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()

    async def _task(i: int):
        async with sem:
            delay = task_delay * (0.5 + random.random())
            await asyncio.sleep(delay)
            return i, delay

    tasks = [asyncio.create_task(_task(i)) for i in range(num_coros)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    durations = []
    completed = 0
    failed = 0
    for r in results:
        if isinstance(r, Exception):
            failed += 1
        else:
            completed += 1
            durations.append(r[1])

    return RunSummary(
        label=f"BoundedGather({num_coros}c_conc{concurrency})",
        config={"num_coros": num_coros, "concurrency": concurrency, "task_delay": task_delay},
        total_tasks=num_coros,
        completed=completed,
        failed=failed,
        cancelled=0,
        durations=durations,
        throughput_per_min=completed / ((time.perf_counter() - t0) / 60) if completed else 0.0,
        wall_clock=time.perf_counter() - t0,
    )


async def _test_rate_limiter(num_requests: int, rate: float) -> RunSummary:
    """Stress-test token bucket rate limiter throughput."""
    limiter = _TokenBucketLimiter("test", rate=rate, capacity=rate)
    t0 = time.perf_counter()
    completed = 0
    wait_times: list[float] = []

    async def _requester():
        nonlocal completed
        w0 = time.perf_counter()
        await limiter.acquire()
        wait_times.append(time.perf_counter() - w0)
        completed += 1

    tasks = [asyncio.create_task(_requester()) for _ in range(num_requests)]
    await asyncio.gather(*tasks)

    return RunSummary(
        label=f"RateLimiter({num_requests}r_rate{rate})",
        config={"num_requests": num_requests, "rate": rate},
        total_tasks=num_requests,
        completed=completed,
        failed=0,
        cancelled=0,
        durations=[],
        throughput_per_min=completed / ((time.perf_counter() - t0) / 60) if completed else 0.0,
        wall_clock=time.perf_counter() - t0,
        semaphore_wait_times=wait_times,
    )


async def _test_circuit_breaker() -> RunSummary:
    """Test CircuitBreaker state transitions under failure bursts."""
    cb = _CircuitBreaker("test", failure_threshold=3, recovery_timeout=2.0, window_seconds=10.0)
    results: list[str] = []
    t0 = time.perf_counter()

    # Phase 1: Fail repeatedly to trip the breaker
    for i in range(5):
        try:
            await cb.call(_always_fail)
            results.append("success")
        except _CircuitBreakerOpenError:
            results.append("open")
        except Exception:
            results.append("failure")

    # Phase 2: Wait for half-open
    await asyncio.sleep(2.1)

    # Phase 3: Succeed to close the breaker
    for i in range(3):
        try:
            await cb.call(_always_ok)
            results.append("success")
        except _CircuitBreakerOpenError:
            results.append("open")
        except Exception:
            results.append("failure")

    return RunSummary(
        label="CircuitBreaker(state_transitions)",
        config={"failure_threshold": 3, "recovery_timeout": 2.0},
        total_tasks=8,
        completed=results.count("success") + results.count("failure"),
        failed=results.count("open"),
        cancelled=0,
        durations=[],
        throughput_per_min=0,
        wall_clock=time.perf_counter() - t0,
    )


async def _always_fail():
    raise RuntimeError("simulated failure")


async def _always_ok():
    return "ok"


# ══════════════════════════════════════════════════════════════════
# Part 2 — Pipeline Simulation (mock LLM/MCP)
# ══════════════════════════════════════════════════════════════════

# Realistic latency profiles based on observed system behavior (seconds)
LLM_LATENCY = {
    "fast":   (0.5, 2.0),
    "plan":   (2.0, 8.0),
    "tool":   (0.3, 1.5),
}

MCP_LATENCY = {
    "pubmed_search": (1.0, 5.0),
    "pubmed_fetch":  (0.5, 3.0),
    "clinical":      (1.5, 6.0),
    "fda":           (1.0, 4.0),
}


def _sim_llm(kind: str = "fast") -> float:
    lo, hi = LLM_LATENCY.get(kind, (0.5, 2.0))
    return random.uniform(lo, hi)


def _sim_mcp(kind: str = "pubmed_search") -> float:
    lo, hi = MCP_LATENCY.get(kind, (1.0, 5.0))
    return random.uniform(lo, hi)


class SimulatedReActAgent:
    """Simulates a ReActSearchAgent with realistic latency but no real API calls."""

    def __init__(self, llm_sem: asyncio.Semaphore, api_sem: asyncio.Semaphore):
        self.llm_sem = llm_sem
        self.api_sem = api_sem

    async def execute(self, query: str, max_rounds: int = 5) -> dict:
        rounds = random.randint(1, min(3, max_rounds))
        ref_count = random.randint(2, 8)

        for r in range(rounds):
            async with self.api_sem:
                await asyncio.sleep(_sim_mcp("pubmed_search"))
            async with self.llm_sem:
                await asyncio.sleep(_sim_llm("fast"))
            async with self.llm_sem:
                await asyncio.sleep(_sim_llm("plan"))
            if r == 0 and random.random() < 0.6:
                break

        return {
            "synthesis": f"[sim] Evidence for: {query[:80]}... ({ref_count} refs)",
            "sufficient": True,
            "follow_up_queries": [],
        }

    async def execute_trial(self, trial_name: str, sub_queries: list[str],
                            max_rounds: int = 5) -> dict:
        rounds = random.randint(2, min(4, max_rounds))
        ref_count = random.randint(4, 15)

        for sq_idx in range(min(len(sub_queries), 3)):
            for r in range(min(rounds, 2)):
                async with self.api_sem:
                    await asyncio.sleep(_sim_mcp("pubmed_search"))
                async with self.llm_sem:
                    await asyncio.sleep(_sim_llm("fast"))

        async with self.llm_sem:
            await asyncio.sleep(_sim_llm("plan"))

        return {
            "synthesis": f"[sim] Trial {trial_name}: {len(sub_queries)} sub-queries, {ref_count} refs",
            "sufficient": True,
            "follow_up_queries": [],
        }


async def _run_simulated_pipeline(
    num_queries: int,
    num_trials: int,
    worker_concurrency: int,
    llm_concurrency: int,
    api_concurrency: int,
    max_rounds: int = 5,
    pipeline_timeout: float = 300.0,
) -> RunSummary:
    """Run a simulated full pipeline with mock agents."""
    llm_sem = asyncio.Semaphore(llm_concurrency)
    api_sem = asyncio.Semaphore(api_concurrency)
    agent = SimulatedReActAgent(llm_sem, api_sem)

    queue = _SimpleTaskQueue()
    done = asyncio.Event()
    metrics: list[TaskMetrics] = []
    queue_depths: list[int] = []
    t0 = time.perf_counter()

    flat_queries = [
        "endometrial cancer AND adjuvant radiotherapy AND survival",
        "cervical cancer AND pembrolizumab AND overall survival",
        "ovarian cancer AND PARP inhibitor AND maintenance",
        "endometrial cancer AND molecular classification AND prognosis",
        "gynecologic cancer AND immunotherapy AND adverse events",
        "endometrial cancer AND sentinel lymph node AND detection rate",
        "cervical cancer AND chemoradiation AND cisplatin VS carboplatin",
        "ovarian cancer AND bevacizumab AND progression-free survival",
        "endometrial cancer AND fertility sparing AND progestin",
        "gynecologic cancer AND frailty AND treatment tolerance",
    ]
    trial_configs = [
        ("PORTEC-3", ["PORTEC-3 overall survival", "PORTEC-3 molecular analysis",
                       "PORTEC-3 toxicity"]),
        ("GOG-0258", ["GOG-0258 CRT vs CT survival", "GOG-0258 subgroup analysis"]),
        ("NRG-GY018", ["NRG-GY018 pembrolizumab endometrial", "NRG-GY018 biomarker"]),
        ("RUBY", ["RUBY dostarlimab endometrial survival", "RUBY MMR subgroup"]),
        ("KEYNOTE-775", ["KEYNOTE-775 lenvatinib pembrolizumab endometrial"]),
    ]

    total_tasks = 0
    for i in range(min(num_trials, len(trial_configs))):
        name, subs = trial_configs[i]
        await queue.put(("trial", (name, subs)))
        total_tasks += 1
    for i in range(num_queries):
        q = flat_queries[i % len(flat_queries)]
        await queue.put(("flat", q))
        total_tasks += 1

    async def _worker():
        while not done.is_set():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if done.is_set() and queue.empty():
                    break
                continue
            except asyncio.CancelledError:
                break

            meta_type, payload = item
            if meta_type is None:
                queue.task_done()
                break

            tm = TaskMetrics(
                task_id=f"{meta_type}:{str(payload)[:60]}",
                start_time=time.perf_counter(),
            )
            try:
                if meta_type == "trial":
                    await agent.execute_trial(payload[0], payload[1], max_rounds)
                else:
                    await agent.execute(str(payload), max_rounds)
                tm.end_time = time.perf_counter()
                tm.duration = tm.end_time - tm.start_time
                tm.success = True
            except asyncio.TimeoutError:
                tm.end_time = time.perf_counter()
                tm.duration = tm.end_time - tm.start_time
                tm.error = "timeout"
            except Exception as e:
                tm.end_time = time.perf_counter()
                tm.duration = tm.end_time - tm.start_time
                tm.error = str(e)[:200]

            metrics.append(tm)
            queue_depths.append(queue.qsize())
            queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(worker_concurrency)]

    try:
        await asyncio.wait_for(queue.join(), timeout=pipeline_timeout)
    except asyncio.TimeoutError:
        logger.warning("Pipeline timeout — some tasks incomplete")

    done.set()
    for _ in workers:
        await queue.put((None, None))
    await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.perf_counter() - t0
    durations = [m.duration for m in metrics if m.success]
    return RunSummary(
        label=f"Pipeline(W{worker_concurrency}_L{llm_concurrency}_A{api_concurrency})",
        config={
            "num_queries": num_queries, "num_trials": num_trials,
            "worker_concurrency": worker_concurrency,
            "llm_concurrency": llm_concurrency,
            "api_concurrency": api_concurrency,
            "max_rounds": max_rounds,
            "pipeline_timeout": pipeline_timeout,
        },
        total_tasks=total_tasks,
        completed=sum(1 for m in metrics if m.success),
        failed=sum(1 for m in metrics if m.error and "timeout" not in m.error),
        cancelled=sum(1 for m in metrics if "timeout" in (m.error or "")),
        durations=durations,
        throughput_per_min=len(durations) / (elapsed / 60) if durations else 0.0,
        wall_clock=elapsed,
        queue_depths=queue_depths,
    )


# ══════════════════════════════════════════════════════════════════
# Part 3 — Live Integration Test (requires full project deps)
# ══════════════════════════════════════════════════════════════════


def _ensure_live_deps():
    """Import full project deps. Raises if not available."""
    global AdvancedSearchSystem, run_evidence_update, get_global_semaphores, settings

    # Ensure project root is on sys.path (script is in scripts/ subdirectory)
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    try:
        from local_deep_research import AdvancedSearchSystem
        from local_deep_research.main import run_evidence_update
        from local_deep_research.search_system import (
            GLOBAL_LLM_SEMAPHORE, GLOBAL_API_SEMAPHORE, get_global_semaphores,
        )
        from local_deep_research.config import settings
    except ImportError as e:
        print(f"\n  ERROR: Full project dependencies not available: {e}")
        print(f"  The 'live' and 'concurrent' modes require:")
        print(f"    - langchain_openai")
        print(f"    - Full PathoEBM installation")
        print(f"  Run '--mode primitives' or '--mode simulate' for offline testing.\n")
        sys.exit(1)


async def _run_live_pipeline(test_file: str, model_choice: str = "deepseek",
                             worker_concurrency: Optional[int] = None,
                             silent: bool = False) -> Optional[float]:
    """Run the real pipeline once. Returns wall clock seconds, or None on failure.

    When silent=True, suppresses the per-trial console output.
    """
    _ensure_live_deps()

    if worker_concurrency is not None:
        settings.pipeline.worker_concurrency = worker_concurrency

    import local_deep_research.search_system as ss
    # Force fresh semaphores with current config values
    ss.GLOBAL_LLM_SEMAPHORE = None
    ss.GLOBAL_API_SEMAPHORE = None

    test_path = Path(test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    treatment_context = test_path.read_text(encoding="utf-8")

    # Suppress framework logging during trial
    if silent:
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("local_deep_research").setLevel(logging.CRITICAL)

    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(
            run_evidence_update(treatment_context, model_choice),
            timeout=1200.0,
        )
        elapsed = time.perf_counter() - t0
        return elapsed
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None
    finally:
        # Restore logging
        if silent:
            logging.getLogger().setLevel(logging.WARNING)


@dataclass
class LiveTrialResult:
    """Aggregated results for one (file, workers) configuration."""
    test_file: str
    worker_concurrency: int
    llm_concurrency: int
    api_concurrency: int
    num_trials: int
    successes: int
    failures: int
    wall_times: list[float] = field(default_factory=list)

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.wall_times) if self.wall_times else 0.0

    @property
    def min_s(self) -> float:
        return min(self.wall_times) if self.wall_times else 0.0

    @property
    def max_s(self) -> float:
        return max(self.wall_times) if self.wall_times else 0.0

    @property
    def stdev_s(self) -> float:
        return statistics.stdev(self.wall_times) if len(self.wall_times) >= 2 else 0.0

    def fmt(self, s: float) -> str:
        m, sec = divmod(s, 60)
        return f"{int(m)}m{sec:.0f}s"

    def summary_line(self) -> str:
        if not self.wall_times:
            return f"  W{self.worker_concurrency:>3}  ALL FAILED  ({self.failures}/{self.num_trials})"
        return (
            f"  W{self.worker_concurrency:>3}  "
            f"mean={self.fmt(self.mean_s):>8s}  "
            f"min={self.fmt(self.min_s):>8s}  "
            f"max={self.fmt(self.max_s):>8s}  "
            f"±{self.stdev_s:.0f}s  "
            f"({self.successes}/{self.num_trials} ok)"
        )


async def run_live_sweep(test_files: list[str], model: str,
                         worker_levels: list[int], trials: int,
                         output_file: str) -> list[LiveTrialResult]:
    """Run live pipeline sweep: test_files × worker_levels × trials.

    Writes clean aggregated results to output_file in real time.
    """
    _ensure_live_deps()

    _actual_llm = getattr(settings.pipeline, 'llm_concurrency', 100)
    _actual_api = getattr(settings.pipeline, 'api_concurrency', 10)

    all_results: list[LiveTrialResult] = []
    total_combos = len(test_files) * len(worker_levels)
    combo_idx = 0

    with open(output_file, "w", encoding="utf-8") as out:
        # Header
        out.write("=" * 72 + "\n")
        out.write("  PathoEBM Live Concurrency Sweep Results\n")
        out.write(f"  Started: {datetime.now().isoformat()}\n")
        out.write(f"  Model: {model}\n")
        out.write(f"  LLM Concurrency: {_actual_llm}\n")
        out.write(f"  API Concurrency: {_actual_api}\n")
        out.write(f"  Worker Levels: {worker_levels}\n")
        out.write(f"  Test Files: {[Path(f).name for f in test_files]}\n")
        out.write(f"  Trials per combo: {trials}\n")
        out.write("=" * 72 + "\n\n")

        for tf in test_files:
            fname = Path(tf).name
            print(f"\n{'─' * 60}")
            print(f"  Test File: {fname}")
            print(f"{'─' * 60}")

            out.write(f"{'─' * 60}\n")
            out.write(f"  {fname}\n")
            out.write(f"{'─' * 60}\n")
            out.write(f"  {'Workers':<6} {'Mean':>10} {'Min':>10} {'Max':>10} {'Stdev':>8} {'Rate':>10}\n")
            out.write(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*10}\n")

            for w in worker_levels:
                combo_idx += 1
                wall_times: list[float] = []
                failures = 0

                print(f"  [{combo_idx}/{total_combos}] W={w} ", end="", flush=True)

                for t in range(trials):
                    elapsed = await _run_live_pipeline(
                        tf, model, worker_concurrency=w, silent=True
                    )
                    if elapsed is not None:
                        wall_times.append(elapsed)
                        print(".", end="", flush=True)
                    else:
                        failures += 1
                        print("x", end="", flush=True)

                r = LiveTrialResult(
                    test_file=fname,
                    worker_concurrency=w,
                    llm_concurrency=_actual_llm,
                    api_concurrency=_actual_api,
                    num_trials=trials,
                    successes=trials - failures,
                    failures=failures,
                    wall_times=wall_times,
                )
                all_results.append(r)

                # Print & write
                line = r.summary_line()
                print(f"\n{line}")
                out.write(line + "\n")
                out.flush()

        # Final summary table
        out.write(f"\n{'=' * 72}\n")
        out.write("  CROSS-FILE SUMMARY (by worker count, averaged across test files)\n")
        out.write(f"{'=' * 72}\n")
        out.write(f"  {'Workers':<6} {'Mean':>10} {'Min':>10} {'Max':>10} {'Stdev':>8}\n")
        out.write(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}\n")

        for w in worker_levels:
            matching = [r for r in all_results if r.worker_concurrency == w and r.wall_times]
            if not matching:
                continue
            all_times = []
            for r in matching:
                all_times.extend(r.wall_times)
            mean_s = statistics.mean(all_times)
            min_s = min(all_times)
            max_s = max(all_times)
            std_s = statistics.stdev(all_times) if len(all_times) >= 2 else 0.0

            def _fmt(s):
                m, sec = divmod(s, 60)
                return f"{int(m)}m{sec:.0f}s"

            line = f"  W{w:<5} {_fmt(mean_s):>10} {_fmt(min_s):>10} {_fmt(max_s):>10} ±{std_s:.0f}s"
            print(line)
            out.write(line + "\n")

        out.write(f"\n  Completed: {datetime.now().isoformat()}\n")

    print(f"\n  Results saved to: {output_file}")
    return all_results


async def _run_concurrent_batch(test_file: str, num_jobs: int, max_concurrent: int,
                                 model_choice: str = "deepseek",
                                 silent: bool = True) -> tuple[float, int, int, list[float]]:
    """Run N independent pipeline processes concurrently.

    Each job runs as a real OS subprocess — its own Python interpreter, MCP
    connections, LLM sockets.  This is the server-level throughput test.

    Returns: (wall_clock_s, successes, failures, per_job_durations)
    """
    test_path = Path(test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Locate the runner script relative to THIS script
    runner = Path(__file__).resolve().parent / "run_pipeline.py"
    if not runner.exists():
        raise FileNotFoundError(f"Runner script not found: {runner}")

    sem = asyncio.Semaphore(max_concurrent)

    t0 = time.perf_counter()
    job_durations: list[float] = []
    job_errors: list[str] = []

    async def _job(job_id: int) -> bool:
        j0 = time.perf_counter()
        try:
            async with sem:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(runner),
                    "--input", str(test_path),
                    "--model", model_choice,
                    "--silent",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=Path(__file__).resolve().parent.parent,  # project root
                )
                stdout, stderr = await proc.communicate()
            elapsed = time.perf_counter() - j0
            job_durations.append(elapsed)
            if proc.returncode == 0:
                return True
            else:
                err_preview = stderr.decode(errors="replace")[-300:] if stderr else ""
                job_errors.append(f"Job {job_id}: exit={proc.returncode}  {err_preview}")
                return False
        except asyncio.CancelledError:
            job_errors.append(f"Job {job_id}: cancelled")
            return False
        except Exception as e:
            job_errors.append(f"Job {job_id}: {str(e)[:200]}")
            return False

    tasks = [asyncio.create_task(_job(i)) for i in range(num_jobs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - t0
    completed = sum(1 for r in results if r is True)
    failed = num_jobs - completed

    # Print per-process errors to stderr so they're visible even in --silent mode
    if job_errors:
        for msg in job_errors:
            print(f"  [subprocess] {msg}", file=sys.stderr, flush=True)

    return elapsed, completed, failed, job_durations


@dataclass
class ConcurrentTrialResult:
    """Aggregated results for one (concurrency_level) configuration."""
    concurrency: int
    num_jobs: int
    num_trials: int
    successes: int
    failures: int
    wall_times: list[float] = field(default_factory=list)
    job_times: list[float] = field(default_factory=list)

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.wall_times) if self.wall_times else 0.0

    @property
    def min_s(self) -> float:
        return min(self.wall_times) if self.wall_times else 0.0

    @property
    def max_s(self) -> float:
        return max(self.wall_times) if self.wall_times else 0.0

    @property
    def stdev_s(self) -> float:
        return statistics.stdev(self.wall_times) if len(self.wall_times) >= 2 else 0.0

    @property
    def job_mean_s(self) -> float:
        return statistics.mean(self.job_times) if self.job_times else 0.0

    @property
    def job_stdev_s(self) -> float:
        return statistics.stdev(self.job_times) if len(self.job_times) >= 2 else 0.0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    def fmt(self, s: float) -> str:
        m, sec = divmod(s, 60)
        return f"{int(m)}m{sec:.0f}s"

    def summary_line(self) -> str:
        if not self.wall_times:
            return f"  C{self.concurrency:>3}  ALL FAILED  ({self.failures}/{self.num_trials})"
        return (
            f"  C{self.concurrency:>3}  "
            f"mean={self.fmt(self.mean_s):>8s}  "
            f"min={self.fmt(self.min_s):>8s}  "
            f"max={self.fmt(self.max_s):>8s}  "
            f"±{self.stdev_s:.0f}s  "
            f"(jobs: avg={self.fmt(self.job_mean_s)}s ±{self.job_stdev_s:.0f}s)  "
            f"({self.successes}/{self.num_trials * self.num_jobs} ok)"
        )


async def run_concurrent_sweep(test_files: list[str], model: str,
                                concurrency_levels: list[int], trials: int,
                                output_file: str) -> list[ConcurrentTrialResult]:
    """Run concurrent pipeline sweep: concurrency_levels × trials.

    For each concurrency level C, launches C jobs simultaneously and measures
    how total wall clock degrades. Writes aggregated results to output_file in real time.
    """
    all_results: list[ConcurrentTrialResult] = []
    total_combos = len(test_files) * len(concurrency_levels)
    combo_idx = 0

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("=" * 72 + "\n")
        out.write("  PathoEBM Concurrent Job Sweep Results\n")
        out.write(f"  Started: {datetime.now().isoformat()}\n")
        out.write(f"  Model: {model}\n")
        out.write(f"  Concurrency Levels: {concurrency_levels}\n")
        out.write(f"  Test Files: {[Path(f).name for f in test_files]}\n")
        out.write(f"  Trials per level: {trials}\n")
        out.write("=" * 72 + "\n\n")
        out.flush()  # 立即刷盘，避免崩溃丢头

        for tf in test_files:
            fname = Path(tf).name
            print(f"\n{'─' * 60}")
            print(f"  Test File: {fname}")
            print(f"{'─' * 60}")

            out.write(f"{'─' * 60}\n")
            out.write(f"  {fname}\n")
            out.write(f"{'─' * 60}\n")
            out.write(f"  {'Conc':<6} {'Mean':>10} {'Min':>10} {'Max':>10} "
                      f"{'Stdev':>8} {'Job Avg':>10} {'Success':>8}\n")
            out.write(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} "
                      f"{'-'*8} {'-'*10} {'-'*8}\n")
            out.flush()

            for c in concurrency_levels:
                combo_idx += 1
                wall_times: list[float] = []
                all_job_times: list[float] = []
                total_successes = 0
                total_failures = 0

                print(f"  [{combo_idx}/{total_combos}] C={c} ", end="", flush=True)

                for t in range(trials):
                    try:
                        elapsed, ok, fail, job_durs = await _run_concurrent_batch(
                            tf, num_jobs=c, max_concurrent=c,
                            model_choice=model, silent=True,
                        )
                        wall_times.append(elapsed)
                        all_job_times.extend(job_durs)
                        total_successes += ok
                        total_failures += fail
                        print("." if fail == 0 else "x", end="", flush=True)
                    except Exception as e:
                        total_failures += c
                        print("E", end="", flush=True)
                        out.write(f"  [ERROR] C={c} trial {t+1}/{trials}: {e}\n")
                        out.flush()

                r = ConcurrentTrialResult(
                    concurrency=c,
                    num_jobs=c,
                    num_trials=trials,
                    successes=total_successes,
                    failures=total_failures,
                    wall_times=wall_times,
                    job_times=all_job_times,
                )
                all_results.append(r)

                line = r.summary_line()
                print(f"\n{line}")
                out.write(line + "\n")
                out.flush()

        # Cross-file summary
        out.write(f"\n{'=' * 72}\n")
        out.write("  CROSS-FILE SUMMARY (by concurrency level)\n")
        out.write(f"{'=' * 72}\n")
        out.write(f"  {'Conc':<6} {'Mean':>10} {'Min':>10} {'Max':>10} "
                  f"{'Stdev':>8} {'Job Avg':>10}\n")
        out.write(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} "
                  f"{'-'*8} {'-'*10}\n")

        for c in concurrency_levels:
            matching = [r for r in all_results if r.concurrency == c and r.wall_times]
            if not matching:
                continue
            all_wall = []
            all_job = []
            successes = 0
            failures = 0
            for r in matching:
                all_wall.extend(r.wall_times)
                all_job.extend(r.job_times)
                successes += r.successes
                failures += r.failures
            mean_s = statistics.mean(all_wall)
            min_s = min(all_wall)
            max_s = max(all_wall)
            std_s = statistics.stdev(all_wall) if len(all_wall) >= 2 else 0.0
            job_mean = statistics.mean(all_job) if all_job else 0.0

            def _fmt(s):
                m, sec = divmod(s, 60)
                return f"{int(m)}m{sec:.0f}s"

            line = (f"  C{c:<5} {_fmt(mean_s):>10} {_fmt(min_s):>10} "
                    f"{_fmt(max_s):>10} ±{std_s:.0f}s  {_fmt(job_mean):>10}s  "
                    f"({successes}/{successes+failures} ok)")
            print(line)
            out.write(line + "\n")

        out.write(f"\n  Completed: {datetime.now().isoformat()}\n")

    print(f"\n  Results saved to: {output_file}")
    return all_results


# ══════════════════════════════════════════════════════════════════
# Report formatting
# ══════════════════════════════════════════════════════════════════


def _print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_summary(r: RunSummary):
    print(f"\n  -- {r.label} --")
    print(f"  Config:          {json.dumps(r.config)}")
    print(f"  Tasks:           {r.total_tasks} total, {r.completed} ok, "
          f"{r.failed} failed, {r.cancelled} cancelled")
    print(f"  Success rate:    {r.success_rate:.1%}")
    print(f"  Wall clock:      {r.wall_clock:.1f}s")
    if r.durations:
        print(f"  Latency (ms):    mean={r.mean*1000:.0f}  p50={r.p50*1000:.0f}  "
              f"p95={r.p95*1000:.0f}  p99={r.p99*1000:.0f}  stdev={r.stdev*1000:.0f}")
        print(f"  Throughput:      {r.throughput_per_min:.1f} tasks/min")
    if r.semaphore_wait_times:
        waits = r.semaphore_wait_times
        print(f"  Semaphore wait:  mean={statistics.mean(waits)*1000:.0f}ms  "
              f"max={max(waits)*1000:.0f}ms")
    if r.queue_depths:
        depths = r.queue_depths
        print(f"  Queue depth:     mean={statistics.mean(depths):.1f}  "
              f"max={max(depths)}  min={min(depths)}")


def _print_comparison_table(summaries: list[RunSummary]):
    """Print a comparison table for worker/limit sweeps."""
    if len(summaries) < 2:
        return
    print(f"\n  {'Label':<40} {'Done':>6} {'Fail':>6} {'Rate':>7} {'Wall':>7} "
          f"{'P50':>7} {'P95':>7} {'P99':>7} {'Tput':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for r in summaries:
        print(f"  {r.label:<40} {r.completed:>6} {r.failed:>6} {r.success_rate:>6.1%} "
              f"{r.wall_clock:>6.1f}s {r.p50*1000:>6.0f}ms {r.p95*1000:>6.0f}ms "
              f"{r.p99*1000:>6.0f}ms {r.throughput_per_min:>7.1f}/m")


# ══════════════════════════════════════════════════════════════════
# Main test orchestrator
# ══════════════════════════════════════════════════════════════════


async def run_primitives_sweep(verbose: bool = False) -> list[RunSummary]:
    """Sweep through concurrency primitives to find limits."""
    summaries: list[RunSummary] = []

    _print_header("1. Task Queue — Worker Scaling")
    for w in [1, 3, 5, 8, 10, 15, 20, 30]:
        s = await _test_task_queue(num_tasks=100, num_workers=w, task_delay=(0.01, 0.05))
        summaries.append(s)
        _print_summary(s)
    _print_comparison_table([s for s in summaries if "Queue" in s.label])

    _print_header("2. Semaphore Contention — Limit Scaling")
    for limit in [2, 4, 8, 16, 32, 64]:
        s = await _test_semaphore_contention(num_tasks=200, semaphore_limit=limit, task_delay=0.02)
        summaries.append(s)
        _print_summary(s)
    _print_comparison_table([s for s in summaries if "Semaphore" in s.label])

    _print_header("3. Bounded Gather — Concurrency Scaling")
    for c in [2, 5, 10, 20, 50, 100]:
        s = await _test_bounded_gather(num_coros=200, concurrency=c, task_delay=0.02)
        summaries.append(s)
        _print_summary(s)

    _print_header("4. Token Bucket Rate Limiter")
    for rate in [3, 5, 10, 50, 100, 500]:
        s = await _test_rate_limiter(num_requests=100, rate=rate)
        summaries.append(s)
        _print_summary(s)

    _print_header("5. Circuit Breaker — State Transitions")
    s = await _test_circuit_breaker()
    summaries.append(s)
    print(f"  State sequence: OPEN after 3 failures -> HALF_OPEN after 2s -> CLOSED after success")
    print(f"  Failed (open): {s.failed} (expected: >=1 breaker-open rejections)")

    return summaries


async def run_simulated_sweep(worker_list: list[int],
                              llm_limits: list[int],
                              api_limits: list[int]) -> list[RunSummary]:
    """Sweep through simulated pipeline configurations."""
    summaries: list[RunSummary] = []

    _print_header("A. Worker Concurrency Sweep (LLM=8, API=10)")
    for w in worker_list:
        s = await _run_simulated_pipeline(
            num_queries=20, num_trials=5,
            worker_concurrency=w, llm_concurrency=8, api_concurrency=10,
            pipeline_timeout=600.0,
        )
        summaries.append(s)
        _print_summary(s)
    _print_comparison_table([s for s in summaries if s.label.startswith("Pipeline(W")])

    _print_header("B. LLM Semaphore Sweep (Workers=15, API=10)")
    for l in llm_limits:
        s = await _run_simulated_pipeline(
            num_queries=20, num_trials=5,
            worker_concurrency=15, llm_concurrency=l, api_concurrency=10,
            pipeline_timeout=600.0,
        )
        summaries.append(s)
        _print_summary(s)

    _print_header("C. API Semaphore Sweep (Workers=15, LLM=8)")
    for a in api_limits:
        s = await _run_simulated_pipeline(
            num_queries=20, num_trials=5,
            worker_concurrency=15, llm_concurrency=8, api_concurrency=a,
            pipeline_timeout=600.0,
        )
        summaries.append(s)
        _print_summary(s)

    _print_header("D. Saturation Test — Extreme Load")
    s = await _run_simulated_pipeline(
        num_queries=50, num_trials=10,
        worker_concurrency=30, llm_concurrency=16, api_concurrency=20,
        pipeline_timeout=600.0,
    )
    summaries.append(s)
    _print_summary(s)

    _print_header("E. Stress Test — High Workers, Constrained LLM")
    s = await _run_simulated_pipeline(
        num_queries=30, num_trials=8,
        worker_concurrency=20, llm_concurrency=4, api_concurrency=10,
        pipeline_timeout=600.0,
    )
    summaries.append(s)
    _print_summary(s)

    return summaries


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="PathoEBM Concurrency Stress Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick primitive stress test (zero deps)
  python scripts/stress_test.py --mode primitives

  # Simulated pipeline with worker scaling
  python scripts/stress_test.py --mode simulate --workers 1 5 10 15 20 30

  # Live sweep: 3 test files × 5 worker levels × 10 trials each
  python scripts/stress_test.py --mode live \
    --test-files /root/PathoEBM/local_deep_research/test.md \
                 /root/PathoEBM/local_deep_research/test_2.md \
                 /root/PathoEBM/local_deep_research/test_3.md \
    --workers 15 30 50 75 100 --trials 10

  # Concurrent sweep: 3 concurrency levels × 5 trials each
  python scripts/stress_test.py --mode concurrent \
    --test-files /root/PathoEBM/local_deep_research/test.md \
    --concurrency-levels 5 8 10 --trials 5
        """,
    )
    p.add_argument("--mode", choices=["primitives", "simulate", "live", "concurrent", "all"],
                   default="primitives",
                   help="Test mode: primitives=isolated stress tests (zero deps), "
                        "simulate=pipeline with mock delays, live=real endpoints, "
                        "concurrent=multiple parallel jobs, all=primitives+simulate")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 5, 10, 15, 20, 30],
                   help="Worker concurrency levels to sweep (default: 1 5 10 15 20 30)")
    p.add_argument("--llm-limits", type=int, nargs="+", default=[2, 4, 8, 16],
                   help="LLM semaphore limits to sweep (default: 2 4 8 16)")
    p.add_argument("--api-limits", type=int, nargs="+", default=[2, 5, 10, 20],
                   help="API semaphore limits to sweep (default: 2 5 10 20)")
    p.add_argument("--test-files", type=str, nargs="+",
                   default=["/root/PathoEBM/local_deep_research/test_3.md"],
                   help="Test markdown files for live tests")
    p.add_argument("--trials", type=int, default=10,
                   help="Number of trials per (file, worker) combination in live mode (default: 10)")
    p.add_argument("--output", type=str, default="",
                   help="Write aggregated results to this file (default: auto-generated name)")
    p.add_argument("--num-jobs", type=int, default=3,
                   help="Number of concurrent jobs for 'concurrent' mode (legacy, use --concurrency-levels)")
    p.add_argument("--concurrency-levels", type=int, nargs="+", default=[5, 8, 10],
                   help="Concurrency levels to sweep in concurrent mode (default: 5 8 10)")
    p.add_argument("--model", choices=["local", "deepseek", "gpt"], default="deepseek",
                   help="Model provider for live tests")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable INFO-level logging")
    p.add_argument("--json-output", type=str, default="",
                   help="Write results as JSON to this file")
    return p.parse_args()


async def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("stress_test").setLevel(logging.DEBUG)

    all_summaries: list[RunSummary] = []

    print(f"\n{'#' * 70}")
    print(f"  PathoEBM Concurrency Stress Test")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Mode: {args.mode}")
    print(f"{'#' * 70}")

    if args.mode in ("primitives", "all"):
        all_summaries.extend(await run_primitives_sweep(verbose=args.verbose))

    if args.mode in ("simulate", "all"):
        all_summaries.extend(await run_simulated_sweep(
            args.workers, args.llm_limits, args.api_limits
        ))

    if args.mode == "live":
        # Generate output filename
        out_file = args.output
        if not out_file:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = str(Path(__file__).resolve().parent.parent / f"stress_results_{ts}.txt")

        print(f"\n  Sweep: {len(args.test_files)} files × {len(args.workers)} workers × {args.trials} trials")
        print(f"  Total runs: {len(args.test_files) * len(args.workers) * args.trials}")
        print(f"  Output: {out_file}\n")

        await run_live_sweep(
            test_files=args.test_files,
            model=args.model,
            worker_levels=args.workers,
            trials=args.trials,
            output_file=out_file,
        )

    if args.mode == "concurrent":
        out_file = args.output
        if not out_file:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = str(Path(__file__).resolve().parent.parent / f"concurrent_results_{ts}.txt")

        print(f"\n  Sweep: {len(args.test_files)} files × {len(args.concurrency_levels)} concurrency levels × {args.trials} trials")
        print(f"  Total runs: {len(args.test_files) * len(args.concurrency_levels) * args.trials}")
        print(f"  Output: {out_file}\n")

        await run_concurrent_sweep(
            test_files=args.test_files,
            model=args.model,
            concurrency_levels=args.concurrency_levels,
            trials=args.trials,
            output_file=out_file,
        )

    # ── Final Report ──
    _print_header("FINAL SUMMARY")
    total_ok = sum(r.completed for r in all_summaries)
    total_fail = sum(r.failed for r in all_summaries)
    print(f"  Total scenarios:  {len(all_summaries)}")
    print(f"  Total tasks ok:   {total_ok}")
    print(f"  Total tasks fail: {total_fail}")
    if (total_ok + total_fail) > 0:
        print(f"  Overall success:  {total_ok / (total_ok + total_fail) * 100:.1f}%")

    # ── Bottleneck Analysis ──
    pipeline_runs = [s for s in all_summaries if "Pipeline" in s.label]
    if len(pipeline_runs) >= 2:
        _print_header("BOTTLENECK ANALYSIS")

        # Worker scaling: find saturation point
        worker_runs = sorted(
            [s for s in pipeline_runs if "Pipeline(W" in s.label and "_L8" in s.label],
            key=lambda s: s.config.get("worker_concurrency", 0),
        )
        if len(worker_runs) >= 2:
            print("\n  Throughput vs Worker Count (LLM=8, API=10):")
            prev_tput = 0
            for r in worker_runs:
                w = r.config["worker_concurrency"]
                tput = r.throughput_per_min
                gain = (tput / prev_tput - 1) * 100 if prev_tput > 0 else float('inf')
                marker = " <-- SATURATION" if prev_tput > 0 and gain < 10 else ""
                print(f"    workers={w:>3}: {tput:>7.1f} tasks/min  "
                      f"(+{gain:>5.0f}% vs prev){marker}")
                prev_tput = tput if tput > 0 else prev_tput

        # Semaphore bottleneck
        llm_runs = sorted(
            [s for s in pipeline_runs if "_L" in s.label and "_A10" in s.label
             and "Pipeline(W15" in s.label],
            key=lambda s: s.config.get("llm_concurrency", 0),
        )
        if len(llm_runs) >= 2:
            print("\n  Latency vs LLM Semaphore (Workers=15, API=10):")
            for r in llm_runs:
                l = r.config["llm_concurrency"]
                print(f"    llm_limit={l:>3}: p50={r.p50*1000:>6.0f}ms  "
                      f"p95={r.p95*1000:>6.0f}ms  p99={r.p99*1000:>6.0f}ms  "
                      f"tput={r.throughput_per_min:>7.1f}/m")

    # ── Recommendations ──
    _print_header("RECOMMENDATIONS")
    print("""
  Key findings to act on:

  1. Worker Concurrency: Find where adding workers yields <10% improvement.
     This is your optimal worker count for the current LLM/API limits.

  2. LLM Semaphore: Match to your provider's rate limits.
     - DeepSeek: check API dashboard for RPM/TPM
     - Local vLLM: bound by GPU memory / batch size (~8-32 typical)

  3. API Semaphore: Match to MCP server capacity.
     Default MCP server handles ~10 concurrent connections.

  4. Queue Health: If mean queue depth grows over time, consumers can't
     keep up with producers - you need backpressure.

  5. For live testing: start with --mode simulate to find theoretical
     limits, then run --mode live with your best config to validate.
""")

    # ── JSON output ──
    if args.json_output:
        output = {
            "timestamp": datetime.now().isoformat(),
            "mode": args.mode,
            "scenarios": [
                {
                    "label": r.label,
                    "config": r.config,
                    "total_tasks": r.total_tasks,
                    "completed": r.completed,
                    "failed": r.failed,
                    "cancelled": r.cancelled,
                    "success_rate": r.success_rate,
                    "wall_clock_s": r.wall_clock,
                    "throughput_per_min": r.throughput_per_min,
                    "latency_p50_ms": r.p50 * 1000,
                    "latency_p95_ms": r.p95 * 1000,
                    "latency_p99_ms": r.p99 * 1000,
                    "latency_mean_ms": r.mean * 1000,
                    "latency_stdev_ms": r.stdev * 1000,
                }
                for r in all_summaries
            ],
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON results written to: {args.json_output}")

    print(f"\n{'#' * 70}")
    print(f"  Stress test complete.")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())
