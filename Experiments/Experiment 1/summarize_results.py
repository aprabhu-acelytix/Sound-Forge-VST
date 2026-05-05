#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one or more Experiment 1 result JSON files."
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        help="One or more run_*.json files produced by brain_sandbox.py",
    )
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


def print_run_summary(payload: dict[str, Any], path: Path) -> None:
    summary = payload.get("summary", {})
    config = payload.get("config", {})
    model = payload.get("model", {})
    model_name = model.get("model_name", "unknown")
    prompts_file = Path(config.get("prompts_file", "unknown")).name

    print(f"Run: {path.name}")
    print(f"- Model: {model_name}")
    print(f"- Prompt fixture: {prompts_file}")
    print(f"- Run state: {payload.get('run_state', 'unknown')}")
    print(
        f"- Overall pass: {summary.get('overall_pass_count', 0)}/"
        f"{summary.get('total_runs', 0)} ({format_pct(summary.get('overall_pass_rate'))})"
    )
    print(
        f"- Syntax pass: {summary.get('syntax_pass_count', 0)}/"
        f"{summary.get('total_runs', 0)} ({format_pct(summary.get('syntax_pass_rate'))})"
    )
    print(
        f"- Range pass: {summary.get('range_pass_count', 0)}/"
        f"{summary.get('total_runs', 0)} ({format_pct(summary.get('range_pass_rate'))})"
    )
    latency = summary.get("latency_ms", {})
    print(f"- Mean latency: {format_ms(latency.get('mean'))}")
    print(f"- P95 latency: {format_ms(latency.get('p95'))}")
    print(f"- Model load: {format_ms(payload.get('model', {}).get('model_load_ms'))}")

    failures = summary.get("failure_categories", {})
    if failures:
        print(f"- Failure categories: {json.dumps(failures, sort_keys=True)}")
    else:
        print("- Failure categories: {}")

    failing_cases = [
        result["prompt_id"]
        for result in payload.get("results", [])
        if not result.get("overall_pass", False)
    ]
    if failing_cases:
        print(f"- Failing prompt IDs: {', '.join(failing_cases)}")
    else:
        print("- Failing prompt IDs: none")


def main() -> int:
    args = parse_args()
    payloads = [(path, load_result(path)) for path in args.result_files]

    for index, (path, payload) in enumerate(payloads):
        if index:
            print()
        print_run_summary(payload, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
