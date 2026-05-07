#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one or more Experiment 1.2 result JSON files."
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


def format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def is_experiment_1_1(payload: dict[str, Any]) -> bool:
    return str(payload.get("experiment", "")).startswith("Experiment 1.1")


def normalized_variant_fields(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    if is_experiment_1_1(payload):
        model_name = Path(get_nested(payload, "model", "model_name") or config.get("model_path", "unknown")).name
        return {
            "patch_model_name": model_name,
            "explainer_model_name": "inline-single-pass",
            "patch_contract": "explanation_first",
            "explanation_mode": "single_pass",
            "explanation_runtime_mode": "inline",
            "few_shot_count": 0,
            "prompts_file": Path(config.get("prompts_file", "unknown")).name,
        }

    models = payload.get("models", {})
    return {
        "patch_model_name": Path(get_nested(models, "patch_model", "model_name") or "unknown").name,
        "explainer_model_name": Path(get_nested(models, "explainer_model", "model_name") or "none").name,
        "patch_contract": config.get("patch_contract", "unknown"),
        "explanation_mode": config.get("explanation_mode", "unknown"),
        "explanation_runtime_mode": config.get("explanation_runtime_mode", "unknown"),
        "few_shot_count": config.get("few_shot_count", 0),
        "prompts_file": Path(config.get("prompts_file", "unknown")).name,
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    fields = normalized_variant_fields(payload)
    if is_experiment_1_1(payload):
        latency = summary.get("latency_ms", {})
        value = {
            "run_state": payload.get("run_state", "unknown"),
            "patch_model": fields["patch_model_name"],
            "explainer_model": fields["explainer_model_name"],
            "variant_key": " | ".join(
                [
                    fields["patch_model_name"],
                    f"explainer={fields['explainer_model_name']}",
                    f"contract={fields['patch_contract']}",
                    f"explanation={fields['explanation_mode']}/{fields['explanation_runtime_mode']}",
                    f"shots={fields['few_shot_count']}",
                    f"fixture={fields['prompts_file']}",
                ]
            ),
            "patch_pipeline_pass_rate": summary.get("overall_pass_rate"),
            "raw_range_pass_rate": summary.get("range_pass_rate"),
            "clamp_intervention_rate": None,
            "clamp_expectation_pass_rate": None,
            "sanitized_sparse_exact_match_rate": summary.get("sparse_pass_rate"),
            "latency_ms": {
                "patch_applied": latency,
                "explanation_ready": latency,
                "ui_gap": {
                    "mean": 0.0 if latency.get("mean") is not None else None,
                    "median": 0.0 if latency.get("median") is not None else None,
                    "min": 0.0 if latency.get("min") is not None else None,
                    "max": 0.0 if latency.get("max") is not None else None,
                    "p95": 0.0 if latency.get("p95") is not None else None,
                    "p99": 0.0 if latency.get("p99") is not None else None,
                },
            },
            "queue_backlog_max": {},
            "failure_categories": summary.get("failure_categories", {}),
            "raw_syntax_pass_rate": summary.get("syntax_pass_rate"),
            "raw_schema_pass_rate": summary.get("schema_pass_rate"),
            "raw_key_order_pass_rate": summary.get("key_order_pass_rate"),
            "raw_semantic_pass_rate": summary.get("sparse_pass_rate"),
        }
        return value

    latency = summary.get("latency_ms", {})
    return {
        "run_state": payload.get("run_state", "unknown"),
        "patch_model": fields["patch_model_name"],
        "explainer_model": fields["explainer_model_name"],
        "variant_key": " | ".join(
            [
                fields["patch_model_name"],
                f"explainer={fields['explainer_model_name']}",
                f"contract={fields['patch_contract']}",
                f"explanation={fields['explanation_mode']}/{fields['explanation_runtime_mode']}",
                f"shots={fields['few_shot_count']}",
                f"fixture={fields['prompts_file']}",
            ]
        ),
        "patch_pipeline_pass_rate": summary.get("patch_pipeline_pass_rate"),
        "raw_range_pass_rate": summary.get("raw_range_pass_rate"),
        "clamp_intervention_rate": summary.get("clamp_intervention_rate"),
        "clamp_expectation_pass_rate": summary.get("clamp_expectation_pass_rate"),
        "sanitized_sparse_exact_match_rate": get_nested(summary, "sparse_exact_match_rate", "sanitized"),
        "latency_ms": {
            "patch_applied": latency.get("patch_applied", {}),
            "explanation_ready": latency.get("explanation_ready", {}),
            "ui_gap": latency.get("ui_gap", {}),
        },
        "queue_backlog_max": summary.get("queue_backlog_max", {}),
        "failure_categories": summary.get("failure_categories", {}),
        "raw_syntax_pass_rate": summary.get("raw_syntax_pass_rate"),
        "raw_schema_pass_rate": summary.get("raw_schema_pass_rate"),
        "raw_key_order_pass_rate": summary.get("raw_key_order_pass_rate"),
        "raw_semantic_pass_rate": summary.get("raw_semantic_pass_rate"),
    }


def build_variant_key(payload: dict[str, Any]) -> str:
    fields = normalized_variant_fields(payload)
    return " | ".join(
        [
            fields["patch_model_name"],
            f"explainer={fields['explainer_model_name']}",
            f"contract={fields['patch_contract']}",
            f"explanation={fields['explanation_mode']}/{fields['explanation_runtime_mode']}",
            f"shots={fields['few_shot_count']}",
            f"fixture={fields['prompts_file']}",
        ]
    )


def print_run_summary(path: Path, payload: dict[str, Any]) -> None:
    normalized = normalize_payload(payload)
    patch_applied_mean = get_nested(normalized, "latency_ms", "patch_applied", "mean")
    ui_gap_mean = get_nested(normalized, "latency_ms", "ui_gap", "mean")
    explanation_ready_mean = get_nested(normalized, "latency_ms", "explanation_ready", "mean")

    print(f"Run: {path.name}")
    print(f"- Variant: {normalized['variant_key']}")
    print(f"- Run state: {normalized['run_state']}")
    print(f"- Patch model: {normalized['patch_model']}")
    print(f"- Explainer model: {normalized['explainer_model']}")
    print(
        f"- Patch pipeline pass: {format_pct(normalized.get('patch_pipeline_pass_rate'))}"
    )
    print(
        f"- Raw range pass: {format_pct(normalized.get('raw_range_pass_rate'))}"
    )
    print(f"- Clamp intervention rate: {format_pct(normalized.get('clamp_intervention_rate'))}")
    print(f"- Clamp expectation pass rate: {format_pct(normalized.get('clamp_expectation_pass_rate'))}")
    print(f"- Sparse exact match rate (sanitized): {format_pct(normalized.get('sanitized_sparse_exact_match_rate'))}")
    print(f"- Mean patch applied latency: {format_ms(patch_applied_mean)}")
    print(f"- P95 patch applied latency: {format_ms(get_nested(normalized, 'latency_ms', 'patch_applied', 'p95'))}")
    print(f"- Mean explanation ready latency: {format_ms(explanation_ready_mean)}")
    print(
        f"- Mean UI gap: {format_ms(ui_gap_mean)} ({format_ratio(safe_divide(ui_gap_mean, patch_applied_mean))} overhang)"
    )
    print(f"- P95 UI gap: {format_ms(get_nested(normalized, 'latency_ms', 'ui_gap', 'p95'))}")
    print(f"- Queue backlog max: {json.dumps(normalized.get('queue_backlog_max', {}), sort_keys=True)}")
    print(f"- Failure categories: {json.dumps(normalized.get('failure_categories', {}), sort_keys=True)}")


def aggregate_variants(payloads: list[tuple[Path, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    variants: dict[str, list[dict[str, Any]]] = {}
    for _, payload in payloads:
        key = build_variant_key(payload)
        variants.setdefault(key, []).append(normalize_payload(payload))
    return variants


def aggregate_variant_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    patch_applied_means = [
        get_nested(entry, "latency_ms", "patch_applied", "mean")
        for entry in entries
        if get_nested(entry, "latency_ms", "patch_applied", "mean") is not None
    ]
    patch_applied_p95s = [
        get_nested(entry, "latency_ms", "patch_applied", "p95")
        for entry in entries
        if get_nested(entry, "latency_ms", "patch_applied", "p95") is not None
    ]
    explanation_ready_means = [
        get_nested(entry, "latency_ms", "explanation_ready", "mean")
        for entry in entries
        if get_nested(entry, "latency_ms", "explanation_ready", "mean") is not None
    ]
    ui_gap_means = [
        get_nested(entry, "latency_ms", "ui_gap", "mean")
        for entry in entries
        if get_nested(entry, "latency_ms", "ui_gap", "mean") is not None
    ]
    ui_gap_p95s = [
        get_nested(entry, "latency_ms", "ui_gap", "p95")
        for entry in entries
        if get_nested(entry, "latency_ms", "ui_gap", "p95") is not None
    ]
    patch_pipeline_rates = [
        entry.get("patch_pipeline_pass_rate")
        for entry in entries
        if entry.get("patch_pipeline_pass_rate") is not None
    ]
    clamp_rates = [
        entry.get("clamp_intervention_rate")
        for entry in entries
        if entry.get("clamp_intervention_rate") is not None
    ]
    sparse_exact_rates = [
        entry.get("sanitized_sparse_exact_match_rate")
        for entry in entries
        if entry.get("sanitized_sparse_exact_match_rate") is not None
    ]

    patch_applied_mean = statistics.fmean(patch_applied_means) if patch_applied_means else None
    ui_gap_mean = statistics.fmean(ui_gap_means) if ui_gap_means else None
    return {
        "run_count": len(entries),
        "patch_applied_mean": patch_applied_mean,
        "patch_applied_p95_mean": statistics.fmean(patch_applied_p95s) if patch_applied_p95s else None,
        "explanation_ready_mean": statistics.fmean(explanation_ready_means) if explanation_ready_means else None,
        "ui_gap_mean": ui_gap_mean,
        "ui_gap_p95_mean": statistics.fmean(ui_gap_p95s) if ui_gap_p95s else None,
        "human_overhang_ratio": safe_divide(ui_gap_mean, patch_applied_mean),
        "patch_pipeline_pass_rate": statistics.fmean(patch_pipeline_rates) if patch_pipeline_rates else None,
        "clamp_intervention_rate": statistics.fmean(clamp_rates) if clamp_rates else None,
        "sanitized_sparse_exact_match_rate": statistics.fmean(sparse_exact_rates) if sparse_exact_rates else None,
    }


def print_variant_comparison(variants: dict[str, list[dict[str, Any]]]) -> None:
    if not variants:
        return
    print("Variant Comparison")
    print(
        "variant | runs | patch_applied_mean | patch_applied_p95 | explanation_ready_mean | ui_gap_mean | ui_gap_p95 | overhang | patch_pass | clamp_rate | sparse_exact"
    )
    sorted_items = sorted(
        variants.items(),
        key=lambda item: (
            aggregate_variant_summary(item[1])["patch_applied_mean"] is None,
            aggregate_variant_summary(item[1])["patch_applied_mean"] or float("inf"),
            item[0],
        ),
    )
    for variant_key, entries in sorted_items:
        aggregate = aggregate_variant_summary(entries)
        print(
            " | ".join(
                [
                    variant_key,
                    str(aggregate["run_count"]),
                    format_ms(aggregate["patch_applied_mean"]),
                    format_ms(aggregate["patch_applied_p95_mean"]),
                    format_ms(aggregate["explanation_ready_mean"]),
                    format_ms(aggregate["ui_gap_mean"]),
                    format_ms(aggregate["ui_gap_p95_mean"]),
                    format_ratio(aggregate["human_overhang_ratio"]),
                    format_pct(aggregate["patch_pipeline_pass_rate"]),
                    format_pct(aggregate["clamp_intervention_rate"]),
                    format_pct(aggregate["sanitized_sparse_exact_match_rate"]),
                ]
            )
        )


def main() -> int:
    args = parse_args()
    payloads = [(path, load_result(path)) for path in args.result_files]
    for index, (path, payload) in enumerate(payloads):
        if index:
            print()
        print_run_summary(path, payload)
    print()
    print_variant_comparison(aggregate_variants(payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
