"""检索词效果遥测 — 将每次检索的 PMID 命中数量写入 JSONL 文件做离线分析。

用法:
    from telemetry import QueryPerformance, record_query_performance
    record_query_performance(QueryPerformance(
        query="PORTEC-3 AND endometrial cancer",
        query_type="trial",
        pmids_found=3,
        was_relaxed=False,
        relaxed_query=None,
        timestamp="2025-01-01T00:00:00",
    ))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TELEMETRY_DIR = Path(__file__).resolve().parent
QUERIES_FILE = TELEMETRY_DIR / "queries.jsonl"


@dataclass
class QueryPerformance:
    query: str
    query_type: str       # "trial" | "pico" | "comorbidity" | "flat"
    pmids_found: int
    was_relaxed: bool
    relaxed_query: str | None
    timestamp: str


def record_query_performance(record: QueryPerformance):
    """追加一条检索效果记录到 JSONL 文件。"""
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUERIES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
