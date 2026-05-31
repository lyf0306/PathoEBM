#!/usr/bin/env python3
"""
Non-interactive pipeline CLI for stress testing.
Usage: python scripts/run_pipeline.py --input <test_file> [--model deepseek]
"""
import argparse
import asyncio
import os
import sys
import time

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from local_deep_research.main import run_evidence_update


async def _main():
    parser = argparse.ArgumentParser(description="Run a single PathoEBM pipeline job.")
    parser.add_argument("--input", required=True, help="Path to test file (.md)")
    parser.add_argument("--model", default="deepseek", help="Model provider")
    parser.add_argument("--silent", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    treatment_context = open(args.input, "r", encoding="utf-8").read()

    if not args.silent:
        print(f"[runner] Starting pipeline: {args.input}  model={args.model}")
    t0 = time.perf_counter()

    try:
        await run_evidence_update(treatment_context, args.model)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    if not args.silent:
        print(f"[runner] Done in {elapsed:.1f}s")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(_main())
