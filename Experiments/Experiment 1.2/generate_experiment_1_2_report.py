#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Experiment 1.2 markdown report from run artifacts.")
    parser.add_argument("result_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def is_experiment_1_1(payload: dict[str, Any]) -> bool:
    return str(payload.get("experiment", "")).startswith("Experiment 1.1")


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    if is_experiment_1_1(payload):
        summary = payload.get("summary", {})
        latency = summary.get("latency_ms", {})
        model_name = Path(get_nested(payload, "model", "model_name") or config.get("model_path", "unknown")).name
        fixture = Path(config.get("prompts_file", "unknown")).name
        return {
            "variant_key": f"{model_name} | explainer=inline-single-pass | contract=explanation_first | explanation=single_pass/inline | shots=0 | fixture={fixture}",
            "fixture": fixture,
            "patch_model": model_name,
            "explainer_model": "inline-single-pass",
            "patch_contract": "explanation_first",
            "explanation_mode": "single_pass",
            "few_shot_count": 0,
            "patch_applied_mean": latency.get("mean"),
            "patch_applied_p95": latency.get("p95"),
            "explanation_ready_mean": latency.get("mean"),
            "ui_gap_mean": 0.0 if latency.get("mean") is not None else None,
            "human_overhang_ratio": 0.0 if latency.get("mean") is not None else None,
            "patch_pipeline_pass_rate": summary.get("overall_pass_rate"),
            "raw_range_pass_rate": summary.get("range_pass_rate"),
            "clamp_intervention_rate": None,
            "clamp_expectation_pass_rate": None,
            "sanitized_sparse_exact_match_rate": summary.get("sparse_pass_rate"),
        }

    summary = payload.get("summary", {})
    models = payload.get("models", {})
    patch_model = Path(get_nested(models, "patch_model", "model_name") or "unknown").name
    explainer_model = Path(get_nested(models, "explainer_model", "model_name") or "none").name
    fixture = Path(config.get("prompts_file", "unknown")).name
    patch_contract = config.get("patch_contract", "unknown")
    explanation_mode = config.get("explanation_mode", "unknown")
    explanation_runtime_mode = config.get("explanation_runtime_mode", "unknown")
    few_shot_count = config.get("few_shot_count", 0)
    patch_applied_mean = get_nested(summary, "latency_ms", "patch_applied", "mean")
    ui_gap_mean = get_nested(summary, "latency_ms", "ui_gap", "mean")
    return {
        "variant_key": f"{patch_model} | explainer={explainer_model} | contract={patch_contract} | explanation={explanation_mode}/{explanation_runtime_mode} | shots={few_shot_count} | fixture={fixture}",
        "fixture": fixture,
        "patch_model": patch_model,
        "explainer_model": explainer_model,
        "patch_contract": patch_contract,
        "explanation_mode": explanation_mode,
        "few_shot_count": few_shot_count,
        "patch_applied_mean": patch_applied_mean,
        "patch_applied_p95": get_nested(summary, "latency_ms", "patch_applied", "p95"),
        "explanation_ready_mean": get_nested(summary, "latency_ms", "explanation_ready", "mean"),
        "ui_gap_mean": ui_gap_mean,
        "human_overhang_ratio": safe_divide(ui_gap_mean, patch_applied_mean),
        "patch_pipeline_pass_rate": summary.get("patch_pipeline_pass_rate"),
        "raw_range_pass_rate": summary.get("raw_range_pass_rate"),
        "clamp_intervention_rate": summary.get("clamp_intervention_rate"),
        "clamp_expectation_pass_rate": summary.get("clamp_expectation_pass_rate"),
        "sanitized_sparse_exact_match_rate": get_nested(summary, "sparse_exact_match_rate", "sanitized"),
    }


def select_variant(normalized: list[dict[str, Any]], *, fixture: str, patch_model_contains: str, contract: str | None = None, explanation_mode: str | None = None, few_shot_count: int | None = None) -> dict[str, Any] | None:
    for item in normalized:
        if item["fixture"] != fixture:
            continue
        if patch_model_contains not in item["patch_model"]:
            continue
        if contract is not None and item["patch_contract"] != contract:
            continue
        if explanation_mode is not None and item["explanation_mode"] != explanation_mode:
            continue
        if few_shot_count is not None and item["few_shot_count"] != few_shot_count:
            continue
        return item
    return None


def format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} ms"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def write_report(output: Path, normalized: list[dict[str, Any]]) -> None:
    a0 = select_variant(normalized, fixture="test_a_latency_prompts_explanation_first.json", patch_model_contains="Qwen2.5-7B", contract="explanation_first")
    a1 = select_variant(normalized, fixture="test_a_latency_prompts.json", patch_model_contains="Qwen2.5-7B", contract="sparse_patch_only", explanation_mode="off")
    a2 = select_variant(normalized, fixture="test_a_latency_prompts.json", patch_model_contains="Qwen2.5-7B", contract="sparse_patch_only", explanation_mode="llm")
    a3 = select_variant(normalized, fixture="test_a_latency_prompts.json", patch_model_contains="Qwen2.5-7B", contract="compact_delta", explanation_mode="off")
    b1 = select_variant(normalized, fixture="test_b_numeric_pressure_prompts.json", patch_model_contains="Qwen2.5-7B", contract="sparse_patch_only")
    b2 = select_variant(normalized, fixture="test_b_numeric_pressure_prompts.json", patch_model_contains="Mistral-7B", contract="sparse_patch_only")
    c0 = select_variant(normalized, fixture="test_c_sparse_ground_truth_prompts.json", patch_model_contains="Qwen2.5-3B", contract="sparse_patch_only", few_shot_count=0)
    c1 = select_variant(normalized, fixture="test_c_sparse_ground_truth_prompts.json", patch_model_contains="Qwen2.5-3B", contract="sparse_patch_only", few_shot_count=2)
    c2 = select_variant(normalized, fixture="test_c_sparse_ground_truth_prompts.json", patch_model_contains="Qwen2.5-3B", contract="sparse_patch_only", few_shot_count=3)

    lines = [
        "# Experiment 1.2 Report",
        "",
        "## Summary",
        "",
        "Experiment 1.2 evaluated whether patch-first generation, clamp-aware middleware, compact delta output, and few-shot prompting improve the Sound Forge Brain control path enough to justify the move toward the eventual JUCE handoff.",
        "",
        "## Test A: Latency And Human Overhang",
        "",
        "Variant | Patch Applied | Explanation Ready | UI Gap | Human Overhang | Patch Pass",
        "--- | --- | --- | --- | --- | ---",
    ]
    for label, item in [("A0", a0), ("A1", a1), ("A2", a2), ("A3", a3)]:
        if item is None:
            continue
        lines.append(
            f"{label} | {format_ms(item['patch_applied_mean'])} | {format_ms(item['explanation_ready_mean'])} | {format_ms(item['ui_gap_mean'])} | {format_ratio(item['human_overhang_ratio'])} | {format_pct(item['patch_pipeline_pass_rate'])}"
        )

    lines.extend(
        [
            "",
            "## Test B: Numeric Pressure And Clamping",
            "",
            "Variant | Raw Range Pass | Clamp Intervention | Clamp Expectation Pass",
            "--- | --- | --- | ---",
        ]
    )
    for label, item in [("B1", b1), ("B2", b2)]:
        if item is None:
            continue
        lines.append(
            f"{label} | {format_pct(item['raw_range_pass_rate'])} | {format_pct(item['clamp_intervention_rate'])} | {format_pct(item['clamp_expectation_pass_rate'])}"
        )

    lines.extend(
        [
            "",
            "## Test C: Sparse Salvage",
            "",
            "Variant | Sparse Exact Match | Patch Pass",
            "--- | --- | ---",
        ]
    )
    for label, item in [("C0", c0), ("C1", c1), ("C2", c2)]:
        if item is None:
            continue
        lines.append(
            f"{label} | {format_pct(item['sanitized_sparse_exact_match_rate'])} | {format_pct(item['patch_pipeline_pass_rate'])}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Test A human overhang: A0={format_ratio(a0['human_overhang_ratio']) if a0 else 'n/a'}, A1={format_ratio(a1['human_overhang_ratio']) if a1 else 'n/a'}, A2={format_ratio(a2['human_overhang_ratio']) if a2 else 'n/a'}, A3={format_ratio(a3['human_overhang_ratio']) if a3 else 'n/a'}.",
            f"- Test B clamp comparison: Qwen 7B clamp intervention={format_pct(b1['clamp_intervention_rate']) if b1 else 'n/a'}, Mistral 7B clamp intervention={format_pct(b2['clamp_intervention_rate']) if b2 else 'n/a'}.",
            f"- Test C sparse salvage: zero-shot={format_pct(c0['sanitized_sparse_exact_match_rate']) if c0 else 'n/a'}, 2-shot={format_pct(c1['sanitized_sparse_exact_match_rate']) if c1 else 'n/a'}, 3-shot={format_pct(c2['sanitized_sparse_exact_match_rate']) if c2 else 'n/a'}.",
            "",
            "## Notes",
            "",
            "- A0 is normalized from the Experiment 1.1 explanation-first contract. Its UI gap is treated as zero because machine application cannot occur until the full explanation-first payload exists.",
            "- Sparse exact match is still the decisive metric for Test C because safe apply alone can hide raw model delta failures.",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    normalized = [normalize_payload(load_result(path)) for path in args.result_files]
    write_report(args.output, normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
