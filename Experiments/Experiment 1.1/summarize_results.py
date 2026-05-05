#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one or more Experiment 1.1 result JSON files."
    )
    parser.add_argument("result_files", nargs="+", type=Path)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} ms"


def main() -> int:
    args = parse_args()
    for index, path in enumerate(args.result_files):
        payload = load_result(path)
        summary = payload.get("summary", {})
        config = payload.get("config", {})
        model = payload.get("model", {})
        if index:
            print()
        print(f"Run: {path.name}")
        print(f"- Model: {model.get('model_name', 'unknown')}")
        print(f"- Prompt fixture: {Path(config.get('prompts_file', 'unknown')).name}")
        print(f"- Run state: {payload.get('run_state', 'unknown')}")
        print(
            f"- Overall pass: {summary.get('overall_pass_count', 0)}/{summary.get('total_runs', 0)} ({format_pct(summary.get('overall_pass_rate'))})"
        )
        print(
            f"- Explanation pass: {summary.get('explanation_pass_count', 0)}/{summary.get('total_runs', 0)} ({format_pct(summary.get('explanation_pass_rate'))})"
        )
        print(
            f"- Range pass: {summary.get('range_pass_count', 0)}/{summary.get('total_runs', 0)} ({format_pct(summary.get('range_pass_rate'))})"
        )
        print(f"- Mean latency: {format_ms(summary.get('latency_ms', {}).get('mean'))}")
        print(f"- P95 latency: {format_ms(summary.get('latency_ms', {}).get('p95'))}")
        print(f"- Mean parse latency: {format_ms(summary.get('parse_latency_ms', {}).get('mean'))}")
        print(f"- Mean validation latency: {format_ms(summary.get('validation_latency_ms', {}).get('mean'))}")
        print(f"- Mean sparse latency: {format_ms(summary.get('sparse_latency_ms', {}).get('mean'))}")
        print(f"- Mean changed params: {summary.get('changed_param_count', {}).get('mean', 'n/a')}")
        print(f"- Mean null ratio: {summary.get('null_ratio', {}).get('mean', 'n/a')}")
        print(f"- All-null no-op count: {summary.get('all_null_noop_count', 0)}")
        print(f"- Failure categories: {json.dumps(summary.get('failure_categories', {}), sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
