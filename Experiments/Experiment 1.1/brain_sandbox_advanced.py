#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import shutil
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    import llama_cpp
    from llama_cpp import Llama, LlamaGrammar
except ImportError as exc:  # pragma: no cover - depends on local install
    raise SystemExit(
        "llama-cpp-python is required. Install it with 'pip install -r "
        "\"Experiments/Experiment 1.1/requirements.txt\"'."
    ) from exc


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = EXPERIMENT_ROOT / "prompts" / "system_prompt_v1_1.txt"
DEFAULT_GRAMMAR_FILE = EXPERIMENT_ROOT / "grammars" / "advanced_patch_v1.gbnf"
DEFAULT_SCHEMA_FILE = EXPERIMENT_ROOT / "schemas" / "advanced_patch_v1.json"
DEFAULT_PROMPTS_FILE = EXPERIMENT_ROOT / "fixtures" / "baseline_prompts.json"
DEFAULT_RESULTS_DIR = EXPERIMENT_ROOT / "results"
DEFAULT_LOGS_DIR = EXPERIMENT_ROOT / "logs"


@dataclass
class PromptCase:
    case_id: str
    prompt: str
    tags: list[str]
    current_patch_context: dict[str, Any] | None


class StatusReporter:
    def __init__(self, status_file: Path | None) -> None:
        self.status_file = status_file
        if self.status_file is not None:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            self.status_file.write_text("", encoding="utf-8")

    def emit(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{timestamp} | {message}"
        print(line, flush=True)
        if self.status_file is not None:
            with self.status_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the advanced Sound Forge Brain Sandbox against a local GGUF model."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a GGUF model.")
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--grammar-file",
        type=Path,
        default=DEFAULT_GRAMMAR_FILE,
        help="Path to the GBNF grammar file.",
    )
    parser.add_argument(
        "--schema-file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Path to the JSON schema file.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="Path to the prompt fixture JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for batch result JSON artifacts.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory for latency CSV artifacts.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Optional log file updated as the run progresses.",
    )
    parser.add_argument(
        "--chat-format",
        default=None,
        help="Optional llama.cpp chat format override for models missing metadata.",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window size.")
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="CPU thread count. Defaults to llama.cpp runtime behavior.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for the most deterministic run.",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.0,
        help="Repeat penalty for decoding.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=640,
        help="Maximum completion tokens per prompt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of prompt cases to execute.",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="Run only specific prompt IDs. Repeat the flag to include more than one.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt_cases(path: Path, selected_ids: set[str], limit: int | None) -> list[PromptCase]:
    payload = load_json(path)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Prompt fixture file must contain a 'cases' list: {path}")

    loaded_cases: list[PromptCase] = []
    for item in cases:
        case_id = item.get("id")
        prompt = item.get("prompt")
        tags = item.get("tags", [])
        current_patch_context = item.get("current_patch_context")

        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Every prompt case needs a non-empty string 'id': {item!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Every prompt case needs a non-empty string 'prompt': {item!r}")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"Prompt case 'tags' must be a list of strings: {item!r}")
        if current_patch_context is not None and not isinstance(current_patch_context, dict):
            raise ValueError(
                f"Prompt case 'current_patch_context' must be an object when present: {item!r}"
            )

        if selected_ids and case_id not in selected_ids:
            continue

        loaded_cases.append(
            PromptCase(
                case_id=case_id,
                prompt=prompt.strip(),
                tags=tags,
                current_patch_context=current_patch_context,
            )
        )

    if limit is not None:
        loaded_cases = loaded_cases[:limit]

    if not loaded_cases:
        raise ValueError("No prompt cases matched the selected filters.")

    return loaded_cases


def load_schema(path: Path) -> dict[str, Any]:
    schema = load_json(path)
    if not isinstance(schema, dict):
        raise ValueError("Schema root must be an object.")
    if schema.get("type") != "object":
        raise ValueError("Schema root type must be 'object'.")
    if not isinstance(schema.get("properties"), dict):
        raise ValueError("Schema must define a 'properties' object.")
    if not isinstance(schema.get("required"), list):
        raise ValueError("Schema must define a 'required' list.")
    return schema


def validate_gbnf_text(grammar_text: str) -> None:
    rule_name_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")
    seen_rule_names: set[str] = set()
    root_present = False

    for raw_line in grammar_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "::=" not in line:
            continue

        rule_name = line.split("::=", 1)[0].strip()
        if not rule_name_pattern.fullmatch(rule_name):
            raise ValueError(
                f"Invalid GBNF rule name {rule_name!r}. Use letters, digits, and hyphens only."
            )
        if rule_name in seen_rule_names:
            raise ValueError(f"Duplicate GBNF rule name: {rule_name!r}.")
        seen_rule_names.add(rule_name)
        if rule_name == "root":
            root_present = True

    if not root_present:
        raise ValueError("GBNF grammar must define a 'root' rule.")


def normalize_assistant_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def parse_json_object(raw_output: str) -> tuple[dict[str, Any] | None, list[str]]:
    trimmed = raw_output.strip()
    if not trimmed:
        return None, ["Empty model output."]

    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}."]

    if not isinstance(parsed, dict):
        return None, [f"JSON root must be an object, received {type(parsed).__name__}."]
    return parsed, []


def flatten_leaf_paths(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_leaf_paths(value, path))
        else:
            flattened[path] = value
    return flattened


def format_bytes(byte_count: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(byte_count)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def detect_nvidia_runtime_visible() -> bool:
    return (
        shutil.which("nvidia-smi") is not None
        or Path("/dev/nvidia0").exists()
        or Path("/dev/dxg").exists()
    )


def detect_llama_gpu_support() -> bool | None:
    probe_functions: list[Any] = []
    support_probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if callable(support_probe):
        probe_functions.append(support_probe)

    low_level = getattr(llama_cpp, "llama_cpp", None)
    if low_level is not None:
        low_level_probe = getattr(low_level, "llama_supports_gpu_offload", None)
        if callable(low_level_probe):
            probe_functions.append(low_level_probe)

    for probe in probe_functions:
        try:
            return bool(probe())
        except Exception:
            continue
    return None


def validate_explanation(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, str):
        return [f"explanation must be a string, received {type(value).__name__}."]

    trimmed = value.strip()
    if not trimmed:
        errors.append("explanation must not be empty.")
    if len(trimmed) < 20:
        errors.append("explanation is too short to be meaningful.")
    if len(trimmed) > 280:
        errors.append("explanation is too long; keep reasoning concise.")
    return errors


def validate_node(
    schema_node: dict[str, Any],
    payload_node: Any,
    *,
    path: str,
    key_order_root_expected: list[str] | None = None,
) -> tuple[list[str], list[str], bool]:
    schema_errors: list[str] = []
    range_errors: list[str] = []
    key_order_pass = True

    if schema_node.get("type") != "object":
        raise ValueError(f"Schema node at {path or 'root'} must be an object schema.")
    if not isinstance(payload_node, dict):
        return [f"{path or 'root'} must be an object, received {type(payload_node).__name__}."], [], False

    properties = schema_node.get("properties", {})
    required_keys = schema_node.get("required", [])
    actual_keys = list(payload_node.keys())
    expected_keys = list(properties.keys())

    if path == "" and key_order_root_expected is not None:
        if actual_keys != key_order_root_expected:
            key_order_pass = False
            schema_errors.append(
                f"Root key order mismatch: expected {key_order_root_expected}, received {actual_keys}."
            )

    missing = [key for key in required_keys if key not in payload_node]
    extra = [key for key in actual_keys if key not in properties]
    if missing:
        schema_errors.append(f"{path or 'root'} missing required keys: {', '.join(missing)}.")
    if extra:
        schema_errors.append(f"{path or 'root'} contains unexpected keys: {', '.join(extra)}.")

    for key in expected_keys:
        if key not in payload_node:
            continue

        child_schema = properties[key]
        child_value = payload_node[key]
        child_path = f"{path}.{key}" if path else key

        if child_schema.get("type") == "object":
            child_schema_errors, child_range_errors, child_key_order = validate_node(
                child_schema,
                child_value,
                path=child_path,
                key_order_root_expected=None,
            )
            schema_errors.extend(child_schema_errors)
            range_errors.extend(child_range_errors)
            key_order_pass = key_order_pass and child_key_order
            continue

        allowed_types = child_schema.get("type")
        if isinstance(allowed_types, list):
            type_set = set(allowed_types)
        elif isinstance(allowed_types, str):
            type_set = {allowed_types}
        else:
            schema_errors.append(f"{child_path} has unsupported schema type declaration.")
            continue

        if child_value is None:
            if "null" not in type_set:
                schema_errors.append(f"{child_path} may not be null.")
            continue

        if "string" in type_set and isinstance(child_value, str):
            enum_values = child_schema.get("enum")
            if enum_values and child_value not in enum_values:
                schema_errors.append(
                    f"{child_path} must be one of {enum_values}, received {child_value!r}."
                )
            continue

        if "integer" in type_set and isinstance(child_value, int) and not isinstance(child_value, bool):
            enum_values = child_schema.get("enum")
            if enum_values and child_value not in enum_values:
                schema_errors.append(
                    f"{child_path} must be one of {enum_values}, received {child_value!r}."
                )
            minimum = child_schema.get("minimum")
            maximum = child_schema.get("maximum")
            if minimum is not None and child_value < int(minimum):
                range_errors.append(f"{child_path} is below minimum {minimum}: received {child_value}.")
            if maximum is not None and child_value > int(maximum):
                range_errors.append(f"{child_path} is above maximum {maximum}: received {child_value}.")
            continue

        if "number" in type_set and isinstance(child_value, (int, float)) and not isinstance(child_value, bool):
            numeric_value = float(child_value)
            if not math.isfinite(numeric_value):
                schema_errors.append(f"{child_path} must be finite, received {child_value!r}.")
                continue
            minimum = child_schema.get("minimum")
            maximum = child_schema.get("maximum")
            if minimum is not None and numeric_value < float(minimum):
                range_errors.append(
                    f"{child_path} is below minimum {minimum}: received {numeric_value}."
                )
            if maximum is not None and numeric_value > float(maximum):
                range_errors.append(
                    f"{child_path} is above maximum {maximum}: received {numeric_value}."
                )
            continue

        schema_errors.append(
            f"{child_path} has incompatible type {type(child_value).__name__}; allowed types are {sorted(type_set)}."
        )

    return schema_errors, range_errors, key_order_pass


def extract_sparse_patch(payload: dict[str, Any]) -> dict[str, Any]:
    changed_paths: dict[str, Any] = {}
    ignored_paths: list[str] = []
    changed_blocks: set[str] = set()

    for block_name, block_value in payload.items():
        if block_name == "explanation":
            continue
        if not isinstance(block_value, dict):
            continue

        for leaf_path, leaf_value in flatten_leaf_paths(block_value, block_name).items():
            if leaf_value is None:
                ignored_paths.append(leaf_path)
            else:
                changed_paths[leaf_path] = leaf_value
                changed_blocks.add(block_name)

    total_leaf_count = len(changed_paths) + len(ignored_paths)
    null_ratio = (len(ignored_paths) / total_leaf_count) if total_leaf_count else 0.0
    return {
        "changed_paths": changed_paths,
        "ignored_paths": ignored_paths,
        "changed_param_count": len(changed_paths),
        "null_param_count": len(ignored_paths),
        "changed_block_count": len(changed_blocks),
        "changed_blocks": sorted(changed_blocks),
        "all_null_noop": len(changed_paths) == 0,
        "null_ratio": null_ratio,
    }


def validate_sparse_output(prompt_case: PromptCase, sparse_patch: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if prompt_case.current_patch_context is not None:
        current_patch_paths = flatten_leaf_paths(prompt_case.current_patch_context)
        if sparse_patch["null_param_count"] == 0:
            errors.append(
                "Sparse update returned no null no-op parameters even though current_patch_context was provided."
            )

        for path, value in sparse_patch["changed_paths"].items():
            if path in current_patch_paths and current_patch_paths[path] == value:
                errors.append(
                    f"{path} matches current_patch_context and should be null in a sparse patch."
                )

    if "noop" in prompt_case.tags and not sparse_patch["all_null_noop"]:
        errors.append("No-op prompt should emit null for every synth parameter.")

    return errors


def merge_sparse_patch(current_patch_context: dict[str, Any] | None, sparse_patch: dict[str, Any]) -> dict[str, Any] | None:
    if current_patch_context is None:
        return None
    merged = json.loads(json.dumps(current_patch_context))
    for path, value in sparse_patch["changed_paths"].items():
        cursor = merged
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return merged


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    blend = rank - lower_index
    return lower_value + ((upper_value - lower_value) * blend)


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def classify_record(record: dict[str, Any]) -> str:
    if record["generation_error"]:
        return "generation_error"
    if not record["syntax_pass"]:
        return "syntax_fail"
    if not record["key_order_pass"]:
        return "key_order_fail"
    if not record["schema_pass"]:
        return "schema_fail"
    if not record["range_pass"]:
        return "range_fail"
    if not record["explanation_pass"]:
        return "explanation_fail"
    if not record["sparse_pass"]:
        return "sparse_fail"
    return "pass"


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records if record["latency_ms"] is not None]
    parse_latencies = [record["parse_latency_ms"] for record in records if record["parse_latency_ms"] is not None]
    validation_latencies = [record["validation_latency_ms"] for record in records if record["validation_latency_ms"] is not None]
    sparse_latencies = [record["sparse_latency_ms"] for record in records if record["sparse_latency_ms"] is not None]
    explanation_lengths = [record["explanation_length"] for record in records if record["explanation_length"] is not None]
    null_ratios = [record["null_ratio"] for record in records if record["null_ratio"] is not None]
    changed_param_counts = [record["changed_param_count"] for record in records]
    all_null_noop_count = sum(1 for record in records if record["all_null_noop"])

    summary = {
        "total_runs": len(records),
        "syntax_pass_count": sum(1 for record in records if record["syntax_pass"]),
        "schema_pass_count": sum(1 for record in records if record["schema_pass"]),
        "range_pass_count": sum(1 for record in records if record["range_pass"]),
        "key_order_pass_count": sum(1 for record in records if record["key_order_pass"]),
        "explanation_pass_count": sum(1 for record in records if record["explanation_pass"]),
        "sparse_pass_count": sum(1 for record in records if record["sparse_pass"]),
        "overall_pass_count": sum(1 for record in records if record["overall_pass"]),
        "syntax_pass_rate": (sum(1 for record in records if record["syntax_pass"]) / len(records)) if records else 0.0,
        "schema_pass_rate": (sum(1 for record in records if record["schema_pass"]) / len(records)) if records else 0.0,
        "range_pass_rate": (sum(1 for record in records if record["range_pass"]) / len(records)) if records else 0.0,
        "key_order_pass_rate": (sum(1 for record in records if record["key_order_pass"]) / len(records)) if records else 0.0,
        "explanation_pass_rate": (sum(1 for record in records if record["explanation_pass"]) / len(records)) if records else 0.0,
        "sparse_pass_rate": (sum(1 for record in records if record["sparse_pass"]) / len(records)) if records else 0.0,
        "overall_pass_rate": (sum(1 for record in records if record["overall_pass"]) / len(records)) if records else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "parse_latency_ms": {
            "mean": statistics.fmean(parse_latencies) if parse_latencies else None,
        },
        "validation_latency_ms": {
            "mean": statistics.fmean(validation_latencies) if validation_latencies else None,
        },
        "sparse_latency_ms": {
            "mean": statistics.fmean(sparse_latencies) if sparse_latencies else None,
        },
        "explanation_length_chars": {
            "mean": statistics.fmean(explanation_lengths) if explanation_lengths else None,
            "max": max(explanation_lengths) if explanation_lengths else None,
        },
        "null_ratio": {
            "mean": statistics.fmean(null_ratios) if null_ratios else None,
            "max": max(null_ratios) if null_ratios else None,
        },
        "changed_param_count": {
            "mean": statistics.fmean(changed_param_counts) if changed_param_counts else None,
            "max": max(changed_param_counts) if changed_param_counts else None,
        },
        "all_null_noop_count": all_null_noop_count,
        "failure_categories": {},
    }

    for record in records:
        category = classify_record(record)
        summary["failure_categories"][category] = summary["failure_categories"].get(category, 0) + 1
    return summary


def build_batch_payload(
    *,
    run_id: str,
    run_state: str,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    model_load_ms: float | None,
    results_path: Path,
    latency_path: Path,
    status_path: Path,
) -> dict[str, Any]:
    return {
        "experiment": "Experiment 1.1: The Advanced Brain Sandbox",
        "run_id": run_id,
        "run_state": run_state,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "artifacts": {
            "results_json": str(results_path),
            "latency_csv": str(latency_path),
            "status_log": str(status_path),
        },
        "config": serialize_args(args),
        "model": {
            "model_path": str(args.model_path),
            "model_name": args.model_path.name,
            "model_file_size": format_bytes(args.model_path.stat().st_size),
            "model_load_ms": model_load_ms,
        },
        "progress": {
            "completed_runs": len(records),
            "total_runs": summary["total_runs"],
        },
        "summary": summary,
        "results": records,
    }


def save_latency_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prompt_id",
        "latency_ms",
        "parse_latency_ms",
        "validation_latency_ms",
        "sparse_latency_ms",
        "syntax_pass",
        "schema_pass",
        "range_pass",
        "key_order_pass",
        "explanation_pass",
        "sparse_pass",
        "overall_pass",
        "changed_param_count",
        "null_param_count",
        "null_ratio",
        "changed_block_count",
        "all_null_noop",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens_per_second",
        "finish_reason",
        "generation_error",
        "validation_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "prompt_id": record["prompt_id"],
                    "latency_ms": f"{record['latency_ms']:.3f}" if record["latency_ms"] is not None else "",
                    "parse_latency_ms": f"{record['parse_latency_ms']:.3f}" if record["parse_latency_ms"] is not None else "",
                    "validation_latency_ms": f"{record['validation_latency_ms']:.3f}" if record["validation_latency_ms"] is not None else "",
                    "sparse_latency_ms": f"{record['sparse_latency_ms']:.3f}" if record["sparse_latency_ms"] is not None else "",
                    "syntax_pass": record["syntax_pass"],
                    "schema_pass": record["schema_pass"],
                    "range_pass": record["range_pass"],
                    "key_order_pass": record["key_order_pass"],
                    "explanation_pass": record["explanation_pass"],
                    "sparse_pass": record["sparse_pass"],
                    "overall_pass": record["overall_pass"],
                    "changed_param_count": record["changed_param_count"],
                    "null_param_count": record["null_param_count"],
                    "null_ratio": f"{record['null_ratio']:.6f}" if record["null_ratio"] is not None else "",
                    "changed_block_count": record["changed_block_count"],
                    "all_null_noop": record["all_null_noop"],
                    "prompt_tokens": record["prompt_tokens"] or "",
                    "completion_tokens": record["completion_tokens"] or "",
                    "total_tokens": record["total_tokens"] or "",
                    "tokens_per_second": f"{record['tokens_per_second']:.3f}" if record["tokens_per_second"] is not None else "",
                    "finish_reason": record["finish_reason"] or "",
                    "generation_error": record["generation_error"] or "",
                    "validation_errors": " | ".join(record["validation_errors"]),
                }
            )


def write_batch_artifacts(
    *,
    run_id: str,
    run_state: str,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    model_load_ms: float | None,
    results_path: Path,
    latency_path: Path,
    status_path: Path,
) -> dict[str, Any]:
    summary = build_summary(records)
    payload = build_batch_payload(
        run_id=run_id,
        run_state=run_state,
        args=args,
        records=records,
        summary=summary,
        model_load_ms=model_load_ms,
        results_path=results_path,
        latency_path=latency_path,
        status_path=status_path,
    )
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_latency_csv(latency_path, records)
    return summary


def build_messages(system_prompt: str, prompt_case: PromptCase) -> list[dict[str, str]]:
    user_parts = []
    if prompt_case.current_patch_context is not None:
        user_parts.append("Current patch context (JSON):")
        user_parts.append(json.dumps(prompt_case.current_patch_context, indent=2))
        user_parts.append("")
    user_parts.append("User request:")
    user_parts.append(prompt_case.prompt)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def run_prompt_case(
    llm: Llama,
    grammar: LlamaGrammar,
    schema: dict[str, Any],
    system_prompt: str,
    prompt_case: PromptCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages = build_messages(system_prompt, prompt_case)

    generation_started = perf_counter()
    raw_output = ""
    response: dict[str, Any] | None = None
    generation_error: str | None = None

    try:
        response = llm.create_chat_completion(
            messages=messages,
            grammar=grammar,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            max_tokens=args.max_tokens,
        )
        choice = response["choices"][0]
        raw_output = normalize_assistant_content(choice["message"].get("content"))
    except Exception as exc:  # pragma: no cover - runtime dependent
        generation_error = str(exc)

    generation_ended = perf_counter()
    latency_ms = (generation_ended - generation_started) * 1000.0

    record: dict[str, Any] = {
        "prompt_id": prompt_case.case_id,
        "prompt": prompt_case.prompt,
        "tags": prompt_case.tags,
        "current_patch_context": prompt_case.current_patch_context,
        "latency_ms": latency_ms,
        "parse_latency_ms": None,
        "validation_latency_ms": None,
        "sparse_latency_ms": None,
        "raw_output": raw_output,
        "trimmed_output": raw_output.strip(),
        "parsed_payload": None,
        "merged_patch_preview": None,
        "syntax_pass": False,
        "schema_pass": False,
        "range_pass": False,
        "key_order_pass": False,
        "explanation_pass": False,
        "sparse_pass": False,
        "overall_pass": False,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "tokens_per_second": None,
        "generation_error": generation_error,
        "validation_errors": [],
        "explanation": None,
        "explanation_length": None,
        "changed_paths": {},
        "ignored_paths": [],
        "changed_param_count": 0,
        "null_param_count": 0,
        "changed_block_count": 0,
        "changed_blocks": [],
        "all_null_noop": False,
        "null_ratio": None,
    }

    if response is not None:
        choice = response["choices"][0]
        record["finish_reason"] = choice.get("finish_reason")
        usage = response.get("usage", {})
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["completion_tokens"] = usage.get("completion_tokens")
        record["total_tokens"] = usage.get("total_tokens")
        if record["completion_tokens"] and latency_ms > 0:
            record["tokens_per_second"] = record["completion_tokens"] / (latency_ms / 1000.0)

    if generation_error:
        record["validation_errors"] = [f"Generation error: {generation_error}"]
        return record

    parse_started = perf_counter()
    parsed_payload, syntax_errors = parse_json_object(raw_output)
    parse_ended = perf_counter()
    record["parse_latency_ms"] = (parse_ended - parse_started) * 1000.0
    if syntax_errors:
        record["validation_errors"] = syntax_errors
        return record

    record["syntax_pass"] = True
    record["parsed_payload"] = parsed_payload
    record["explanation"] = parsed_payload.get("explanation")
    if isinstance(record["explanation"], str):
        record["explanation_length"] = len(record["explanation"].strip())

    validation_started = perf_counter()
    schema_errors, range_errors, key_order_pass = validate_node(
        schema,
        parsed_payload,
        path="",
        key_order_root_expected=["explanation", "oscillator-1", "oscillator-2", "filter", "envelope", "effects"],
    )
    explanation_errors = validate_explanation(parsed_payload.get("explanation"))
    validation_ended = perf_counter()
    record["validation_latency_ms"] = (validation_ended - validation_started) * 1000.0

    sparse_started = perf_counter()
    sparse_patch = extract_sparse_patch(parsed_payload)
    sparse_errors = validate_sparse_output(prompt_case, sparse_patch)
    sparse_ended = perf_counter()
    record["sparse_latency_ms"] = (sparse_ended - sparse_started) * 1000.0

    record["changed_paths"] = sparse_patch["changed_paths"]
    record["ignored_paths"] = sparse_patch["ignored_paths"]
    record["changed_param_count"] = sparse_patch["changed_param_count"]
    record["null_param_count"] = sparse_patch["null_param_count"]
    record["changed_block_count"] = sparse_patch["changed_block_count"]
    record["changed_blocks"] = sparse_patch["changed_blocks"]
    record["all_null_noop"] = sparse_patch["all_null_noop"]
    record["null_ratio"] = sparse_patch["null_ratio"]
    record["merged_patch_preview"] = merge_sparse_patch(prompt_case.current_patch_context, sparse_patch)

    record["schema_pass"] = not schema_errors
    record["range_pass"] = not range_errors
    record["key_order_pass"] = key_order_pass
    record["explanation_pass"] = not explanation_errors
    record["sparse_pass"] = not sparse_errors
    record["validation_errors"] = [*schema_errors, *range_errors, *explanation_errors, *sparse_errors]
    record["overall_pass"] = (
        record["syntax_pass"]
        and record["schema_pass"]
        and record["range_pass"]
        and record["key_order_pass"]
        and record["explanation_pass"]
        and record["sparse_pass"]
    )
    return record


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    ensure_file(args.model_path, "Model file")
    ensure_file(args.system_prompt_file, "System prompt file")
    ensure_file(args.grammar_file, "Grammar file")
    ensure_file(args.schema_file, "Schema file")
    ensure_file(args.prompts_file, "Prompt fixture file")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = args.results_dir / f"run_{run_id}.json"
    latency_path = args.logs_dir / f"latency_{run_id}.csv"
    status_path = args.status_file or (args.logs_dir / f"status_{run_id}.log")
    reporter = StatusReporter(status_path)

    reporter.emit("Starting Experiment 1.1: The Advanced Brain Sandbox")
    reporter.emit(f"Run ID: {run_id}")
    reporter.emit(f"Model path: {args.model_path}")
    reporter.emit(f"Model file size: {format_bytes(args.model_path.stat().st_size)}")
    reporter.emit(f"Results JSON: {results_path}")
    reporter.emit(f"Latency CSV: {latency_path}")
    reporter.emit(f"Status log: {status_path}")

    system_prompt = load_text(args.system_prompt_file)
    schema = load_schema(args.schema_file)
    prompt_cases = load_prompt_cases(args.prompts_file, selected_ids=set(args.prompt_id), limit=args.limit)

    reporter.emit(f"Validating grammar file: {args.grammar_file}")
    try:
        grammar_text = args.grammar_file.read_text(encoding="utf-8")
        validate_gbnf_text(grammar_text)
        grammar = LlamaGrammar.from_file(str(args.grammar_file))
    except Exception as exc:
        reporter.emit(f"Grammar validation failed: {exc}")
        summary = write_batch_artifacts(
            run_id=run_id,
            run_state="failed",
            args=args,
            records=[],
            model_load_ms=None,
            results_path=results_path,
            latency_path=latency_path,
            status_path=status_path,
        )
        reporter.emit(f"Results JSON written to: {results_path}")
        reporter.emit(f"Latency CSV written to: {latency_path}")
        reporter.emit(f"Status log written to: {status_path}")
        return 2

    reporter.emit(f"Selected prompt cases: {len(prompt_cases)}")
    if args.n_gpu_layers <= 0:
        reporter.emit("Inference mode: CPU-only (`--n-gpu-layers 0`).")
    else:
        gpu_runtime_visible = detect_nvidia_runtime_visible()
        llama_gpu_support = detect_llama_gpu_support()
        if not gpu_runtime_visible:
            reporter.emit(
                f"Inference mode warning: GPU offload requested with {args.n_gpu_layers} layers, but no Nvidia runtime is visible in this environment."
            )
        elif llama_gpu_support is False:
            reporter.emit(
                f"Inference mode warning: GPU offload requested with {args.n_gpu_layers} layers, but the installed llama.cpp backend does not report GPU offload support."
            )
        else:
            reporter.emit(f"Inference mode: GPU offload requested with {args.n_gpu_layers} layers.")

    llama_kwargs: dict[str, Any] = {
        "model_path": str(args.model_path),
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "seed": args.seed,
        "verbose": False,
    }
    if args.n_threads is not None:
        llama_kwargs["n_threads"] = args.n_threads
    if args.chat_format is not None:
        llama_kwargs["chat_format"] = args.chat_format

    reporter.emit("Stage 1/3: loading the GGUF model into memory.")
    model_load_started = perf_counter()
    llm = Llama(**llama_kwargs)
    model_load_ms = (perf_counter() - model_load_started) * 1000.0
    reporter.emit(f"Stage 1/3 complete: model load finished in {model_load_ms:.2f} ms.")

    records: list[dict[str, Any]] = []
    write_batch_artifacts(
        run_id=run_id,
        run_state="running",
        args=args,
        records=records,
        model_load_ms=model_load_ms,
        results_path=results_path,
        latency_path=latency_path,
        status_path=status_path,
    )

    try:
        for index, prompt_case in enumerate(prompt_cases, start=1):
            reporter.emit(f"Stage 2/3: starting prompt {index}/{len(prompt_cases)} [{prompt_case.case_id}]")
            record = run_prompt_case(
                llm=llm,
                grammar=grammar,
                schema=schema,
                system_prompt=system_prompt,
                prompt_case=prompt_case,
                args=args,
            )
            records.append(record)
            write_batch_artifacts(
                run_id=run_id,
                run_state="running",
                args=args,
                records=records,
                model_load_ms=model_load_ms,
                results_path=results_path,
                latency_path=latency_path,
                status_path=status_path,
            )
            status = "PASS" if record["overall_pass"] else "FAIL"
            error_summary = " | ".join(record["validation_errors"]) if record["validation_errors"] else "-"
            reporter.emit(
                f"Stage 2/3: completed prompt {index}/{len(prompt_cases)} [{prompt_case.case_id}] | {status} | {record['latency_ms']:.2f} ms | finish={record['finish_reason'] or 'n/a'} | changed={record['changed_param_count']} | nulls={record['null_param_count']} | errors={error_summary}"
            )
    except KeyboardInterrupt:
        summary = write_batch_artifacts(
            run_id=run_id,
            run_state="interrupted",
            args=args,
            records=records,
            model_load_ms=model_load_ms,
            results_path=results_path,
            latency_path=latency_path,
            status_path=status_path,
        )
        reporter.emit("Run interrupted by user. Partial artifacts were written before exit.")
        reporter.emit(f"Completed prompts before interruption: {summary['total_runs']}")
        return 130

    summary = write_batch_artifacts(
        run_id=run_id,
        run_state="completed",
        args=args,
        records=records,
        model_load_ms=model_load_ms,
        results_path=results_path,
        latency_path=latency_path,
        status_path=status_path,
    )

    reporter.emit("Stage 3/3: final artifact write complete.")
    reporter.emit(f"Total runs: {summary['total_runs']}")
    reporter.emit(
        f"Overall pass rate: {summary['overall_pass_count']}/{summary['total_runs']} ({summary['overall_pass_rate']:.1%})"
    )
    reporter.emit(
        f"Syntax pass rate: {summary['syntax_pass_count']}/{summary['total_runs']} ({summary['syntax_pass_rate']:.1%})"
    )
    reporter.emit(
        f"Range pass rate: {summary['range_pass_count']}/{summary['total_runs']} ({summary['range_pass_rate']:.1%})"
    )
    reporter.emit(
        f"Explanation pass rate: {summary['explanation_pass_count']}/{summary['total_runs']} ({summary['explanation_pass_rate']:.1%})"
    )
    if summary["latency_ms"]["mean"] is not None:
        reporter.emit(f"Mean latency: {summary['latency_ms']['mean']:.2f} ms")
    if summary["latency_ms"]["p95"] is not None:
        reporter.emit(f"P95 latency: {summary['latency_ms']['p95']:.2f} ms")
    if summary["null_ratio"]["mean"] is not None:
        reporter.emit(f"Mean null ratio: {summary['null_ratio']['mean']:.3f}")
    reporter.emit(f"Results JSON written to: {results_path}")
    reporter.emit(f"Latency CSV written to: {latency_path}")
    reporter.emit(f"Status log written to: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
