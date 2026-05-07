#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import queue
import re
import shutil
import statistics
import sys
import threading
from collections import defaultdict
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
DEFAULT_RESULTS_DIR = EXPERIMENT_ROOT / "results"
DEFAULT_LOGS_DIR = EXPERIMENT_ROOT / "logs"
DEFAULT_PROMPTS_FILE = EXPERIMENT_ROOT / "fixtures" / "test_a_latency_prompts.json"
DEFAULT_PATCH_SPARSE_GRAMMAR_FILE = EXPERIMENT_ROOT / "grammars" / "sparse_patch_only_v1.gbnf"
DEFAULT_PATCH_SPARSE_SCHEMA_FILE = EXPERIMENT_ROOT / "schemas" / "sparse_patch_only_v1.json"
DEFAULT_PATCH_DELTA_GRAMMAR_FILE = EXPERIMENT_ROOT / "grammars" / "compact_delta_patch_v1.gbnf"
DEFAULT_PATCH_DELTA_SCHEMA_FILE = EXPERIMENT_ROOT / "schemas" / "compact_delta_patch_v1.json"
DEFAULT_EXPLANATION_GRAMMAR_FILE = EXPERIMENT_ROOT / "grammars" / "explanation_response_v1.gbnf"
DEFAULT_EXPLANATION_SCHEMA_FILE = EXPERIMENT_ROOT / "schemas" / "explanation_response_v1.json"
DEFAULT_PARAMETER_REGISTRY_FILE = EXPERIMENT_ROOT / "configs" / "parameter_registry_v1.json"

PATCH_CONTRACT_SPARSE = "sparse_patch_only"
PATCH_CONTRACT_COMPACT = "compact_delta"
EXPLANATION_MODE_OFF = "off"
EXPLANATION_MODE_DETERMINISTIC = "deterministic"
EXPLANATION_MODE_LLM = "llm"
EXPLANATION_RUNTIME_ISOLATED = "isolated"
EXPLANATION_RUNTIME_SERIALIZED = "serialized"
PATCH_BLOCK_ORDER = ["oscillator-1", "oscillator-2", "filter", "envelope", "effects"]
SENTINEL = object()


@dataclass
class PromptCase:
    case_id: str
    prompt: str
    tags: list[str]
    stream_id: str
    current_patch_context: dict[str, Any] | None
    expected_changed_paths: list[str] | None
    expected_exact_values: dict[str, Any] | None
    expected_null_paths: list[str] | None
    expected_clamped_paths: list[str] | None
    expected_noop: bool | None


@dataclass
class FewShotExample:
    user: str
    assistant: str
    tags: list[str]


@dataclass
class ModelHandle:
    label: str
    model_path: Path
    llm: Llama
    lock: threading.Lock
    load_ms: float
    shared_with_patch: bool = False


@dataclass
class PipelineJob:
    prompt_case: PromptCase
    record: dict[str, Any]
    submitted_perf: float
    patch_queue_entered_perf: float
    apply_queue_entered_perf: float | None = None
    explanation_queue_entered_perf: float | None = None
    patch_version: int | None = None


class StatusReporter:
    def __init__(self, status_file: Path | None) -> None:
        self.status_file = status_file
        self._lock = threading.Lock()
        if self.status_file is not None:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            self.status_file.write_text("", encoding="utf-8")

    def emit(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{timestamp} | {message}"
        with self._lock:
            print(line, flush=True)
            if self.status_file is not None:
                with self.status_file.open("a", encoding="utf-8") as handle:
                    handle.write(f"{line}\n")


class SharedPipelineState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._max_queue_sizes = {
            "patch_input": 0,
            "apply": 0,
            "explanation": 0,
            "completion": 0,
        }
        self._latest_patch_versions: dict[str, int] = {}

    def record_queue_size(self, queue_name: str, size: int) -> None:
        with self._lock:
            self._max_queue_sizes[queue_name] = max(self._max_queue_sizes[queue_name], size)

    def next_patch_version(self, stream_id: str) -> int:
        with self._lock:
            next_version = self._latest_patch_versions.get(stream_id, 0) + 1
            self._latest_patch_versions[stream_id] = next_version
            return next_version

    def latest_patch_version(self, stream_id: str) -> int:
        with self._lock:
            return self._latest_patch_versions.get(stream_id, 0)

    def max_queue_sizes(self) -> dict[str, int]:
        with self._lock:
            return dict(self._max_queue_sizes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1.2 for patch-only, clamp-aware, async Sound Forge Brain pipelines."
    )
    parser.add_argument(
        "--patch-model-path",
        type=Path,
        required=True,
        help="Path to the patch-generation GGUF model.",
    )
    parser.add_argument(
        "--explainer-model-path",
        type=Path,
        default=None,
        help="Optional path to the explanation GGUF model. Defaults to the patch model when explanation mode is llm.",
    )
    parser.add_argument(
        "--patch-contract",
        choices=[PATCH_CONTRACT_SPARSE, PATCH_CONTRACT_COMPACT],
        default=PATCH_CONTRACT_SPARSE,
        help="Patch output contract to test.",
    )
    parser.add_argument(
        "--explanation-mode",
        choices=[EXPLANATION_MODE_OFF, EXPLANATION_MODE_DETERMINISTIC, EXPLANATION_MODE_LLM],
        default=EXPLANATION_MODE_OFF,
        help="How to produce the UI explanation after apply.",
    )
    parser.add_argument(
        "--explanation-runtime-mode",
        choices=[EXPLANATION_RUNTIME_ISOLATED, EXPLANATION_RUNTIME_SERIALIZED],
        default=EXPLANATION_RUNTIME_ISOLATED,
        help="Use a dedicated explainer model instance or serialize explanations through the patch model instance.",
    )
    parser.add_argument(
        "--patch-system-prompt-file",
        type=Path,
        default=None,
        help="Optional override for the patch system prompt.",
    )
    parser.add_argument(
        "--explanation-system-prompt-file",
        type=Path,
        default=None,
        help="Optional override for the explanation system prompt.",
    )
    parser.add_argument(
        "--few-shot-file",
        type=Path,
        default=None,
        help="Optional JSON file containing user/assistant few-shot examples for the patch model.",
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=0,
        help="Number of few-shot user/assistant examples to prepend before the real user request.",
    )
    parser.add_argument(
        "--patch-grammar-file",
        type=Path,
        default=None,
        help="Optional override for the patch grammar file.",
    )
    parser.add_argument(
        "--patch-schema-file",
        type=Path,
        default=None,
        help="Optional override for the patch schema file.",
    )
    parser.add_argument(
        "--explanation-grammar-file",
        type=Path,
        default=DEFAULT_EXPLANATION_GRAMMAR_FILE,
        help="Grammar file for the explanation worker when explanation mode is llm.",
    )
    parser.add_argument(
        "--explanation-schema-file",
        type=Path,
        default=DEFAULT_EXPLANATION_SCHEMA_FILE,
        help="Schema file for the explanation worker when explanation mode is llm.",
    )
    parser.add_argument(
        "--parameter-registry-file",
        type=Path,
        default=DEFAULT_PARAMETER_REGISTRY_FILE,
        help="Canonical registry for parameter validation and clamping.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="Prompt fixture JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for run result JSON artifacts.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory for status, CSV, and clamp logs.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Optional explicit status log path.",
    )
    parser.add_argument("--patch-chat-format", default=None, help="Optional patch chat format override.")
    parser.add_argument(
        "--explainer-chat-format",
        default=None,
        help="Optional explainer chat format override.",
    )
    parser.add_argument("--patch-n-ctx", type=int, default=4096, help="Patch model context window.")
    parser.add_argument("--explainer-n-ctx", type=int, default=2048, help="Explainer model context window.")
    parser.add_argument(
        "--patch-n-threads",
        type=int,
        default=None,
        help="Patch model CPU thread count.",
    )
    parser.add_argument(
        "--explainer-n-threads",
        type=int,
        default=None,
        help="Explainer model CPU thread count.",
    )
    parser.add_argument(
        "--patch-n-gpu-layers",
        type=int,
        default=0,
        help="Patch model GPU-offloaded layer count.",
    )
    parser.add_argument(
        "--explainer-n-gpu-layers",
        type=int,
        default=0,
        help="Explainer model GPU-offloaded layer count.",
    )
    parser.add_argument(
        "--patch-temperature",
        type=float,
        default=0.0,
        help="Patch generation sampling temperature.",
    )
    parser.add_argument(
        "--explainer-temperature",
        type=float,
        default=0.0,
        help="Explanation generation sampling temperature.",
    )
    parser.add_argument("--patch-top-p", type=float, default=1.0, help="Patch generation top-p.")
    parser.add_argument(
        "--explainer-top-p",
        type=float,
        default=1.0,
        help="Explanation generation top-p.",
    )
    parser.add_argument(
        "--patch-repeat-penalty",
        type=float,
        default=1.0,
        help="Patch generation repeat penalty.",
    )
    parser.add_argument(
        "--explainer-repeat-penalty",
        type=float,
        default=1.0,
        help="Explanation generation repeat penalty.",
    )
    parser.add_argument(
        "--patch-max-tokens",
        type=int,
        default=384,
        help="Maximum patch completion tokens per prompt.",
    )
    parser.add_argument(
        "--explainer-max-tokens",
        type=int,
        default=96,
        help="Maximum explanation completion tokens per prompt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on prompt cases.",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="Run only specific prompt IDs. Repeat to include more than one.",
    )
    parser.add_argument(
        "--worker-queue-size",
        type=int,
        default=64,
        help="Bounded queue size for the mock async worker pipeline.",
    )
    return parser.parse_args()


def ensure_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_copy_json(value: Any) -> Any:
    return copy.deepcopy(value)


def resolve_patch_contract_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.patch_contract == PATCH_CONTRACT_SPARSE:
        default_grammar = DEFAULT_PATCH_SPARSE_GRAMMAR_FILE
        default_schema = DEFAULT_PATCH_SPARSE_SCHEMA_FILE
    else:
        default_grammar = DEFAULT_PATCH_DELTA_GRAMMAR_FILE
        default_schema = DEFAULT_PATCH_DELTA_SCHEMA_FILE
    return args.patch_grammar_file or default_grammar, args.patch_schema_file or default_schema


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
        stream_id = item.get("stream_id", case_id)
        current_patch_context = item.get("current_patch_context")
        expected_changed_paths = item.get("expected_changed_paths")
        expected_exact_values = item.get("expected_exact_values")
        expected_null_paths = item.get("expected_null_paths")
        expected_clamped_paths = item.get("expected_clamped_paths")
        expected_noop = item.get("expected_noop")

        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Every prompt case needs a non-empty string 'id': {item!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Every prompt case needs a non-empty string 'prompt': {item!r}")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"Prompt case 'tags' must be a list of strings: {item!r}")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError(f"Prompt case 'stream_id' must be a non-empty string when present: {item!r}")
        if current_patch_context is not None and not isinstance(current_patch_context, dict):
            raise ValueError(
                f"Prompt case 'current_patch_context' must be an object when present: {item!r}"
            )
        if expected_changed_paths is not None and (
            not isinstance(expected_changed_paths, list)
            or not all(isinstance(path_value, str) for path_value in expected_changed_paths)
        ):
            raise ValueError(
                f"Prompt case 'expected_changed_paths' must be a list of strings when present: {item!r}"
            )
        if expected_exact_values is not None and not isinstance(expected_exact_values, dict):
            raise ValueError(
                f"Prompt case 'expected_exact_values' must be an object when present: {item!r}"
            )
        if expected_null_paths is not None and (
            not isinstance(expected_null_paths, list)
            or not all(isinstance(path_value, str) for path_value in expected_null_paths)
        ):
            raise ValueError(
                f"Prompt case 'expected_null_paths' must be a list of strings when present: {item!r}"
            )
        if expected_clamped_paths is not None and (
            not isinstance(expected_clamped_paths, list)
            or not all(isinstance(path_value, str) for path_value in expected_clamped_paths)
        ):
            raise ValueError(
                f"Prompt case 'expected_clamped_paths' must be a list of strings when present: {item!r}"
            )
        if expected_noop is not None and not isinstance(expected_noop, bool):
            raise ValueError(f"Prompt case 'expected_noop' must be a bool when present: {item!r}")

        if selected_ids and case_id not in selected_ids:
            continue

        loaded_cases.append(
            PromptCase(
                case_id=case_id,
                prompt=prompt.strip(),
                tags=tags,
                stream_id=stream_id,
                current_patch_context=deep_copy_json(current_patch_context),
                expected_changed_paths=list(expected_changed_paths) if expected_changed_paths is not None else None,
                expected_exact_values=deep_copy_json(expected_exact_values),
                expected_null_paths=list(expected_null_paths) if expected_null_paths is not None else None,
                expected_clamped_paths=list(expected_clamped_paths) if expected_clamped_paths is not None else None,
                expected_noop=expected_noop,
            )
        )

    if limit is not None:
        loaded_cases = loaded_cases[:limit]

    if not loaded_cases:
        raise ValueError("No prompt cases matched the selected filters.")
    return loaded_cases


def load_few_shot_examples(path: Path | None, count: int) -> list[FewShotExample]:
    if count <= 0:
        return []
    if path is None:
        raise ValueError("--few-shot-count requires --few-shot-file.")

    payload = load_json(path)
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Few-shot file must contain a non-empty 'examples' list: {path}")

    loaded_examples: list[FewShotExample] = []
    for item in examples[:count]:
        user = item.get("user")
        assistant = item.get("assistant")
        tags = item.get("tags", [])
        if not isinstance(user, str) or not user.strip():
            raise ValueError(f"Few-shot example is missing a non-empty 'user' string: {item!r}")
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError(f"Few-shot example is missing a non-empty 'assistant' string: {item!r}")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"Few-shot example 'tags' must be a list of strings: {item!r}")
        loaded_examples.append(
            FewShotExample(user=user.strip(), assistant=assistant.strip(), tags=tags)
        )

    if len(loaded_examples) < count:
        raise ValueError(
            f"Few-shot file only contains {len(loaded_examples)} examples, but {count} were requested."
        )
    return loaded_examples


def load_parameter_registry(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Parameter registry root must be an object.")
    parameters = payload.get("parameters")
    parameter_order = payload.get("parameter_order")
    block_order = payload.get("block_order")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Parameter registry must contain a non-empty 'parameters' object.")
    if not isinstance(parameter_order, list) or not parameter_order:
        raise ValueError("Parameter registry must contain a non-empty 'parameter_order' list.")
    if not isinstance(block_order, list) or not block_order:
        raise ValueError("Parameter registry must contain a non-empty 'block_order' list.")
    for path_key in parameter_order:
        if path_key not in parameters:
            raise ValueError(f"Parameter order references unknown path: {path_key}")
    return payload


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


def assign_nested_path(root: dict[str, Any], path: str, value: Any) -> None:
    cursor = root
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def build_full_null_sparse_patch(registry: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {block_name: {} for block_name in registry["block_order"]}
    for path in registry["parameter_order"]:
        assign_nested_path(payload, path, None)
    return payload


def build_sparse_patch_preview(changed_paths: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    payload = build_full_null_sparse_patch(registry)
    for path in registry["parameter_order"]:
        if path in changed_paths:
            assign_nested_path(payload, path, changed_paths[path])
    return payload


def build_compact_delta_preview(changed_paths: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    for path in registry["parameter_order"]:
        if path in changed_paths:
            changes.append({"path": path, "value": changed_paths[path]})
    return {"changes": changes}


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


def summarize_metric(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def validate_explanation_text(value: Any) -> list[str]:
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


def validate_explanation_payload(payload: Any, schema: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return [f"Explanation payload root must be an object, received {type(payload).__name__}."]

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    actual_keys = list(payload.keys())
    expected_keys = list(properties.keys())
    errors: list[str] = []
    if actual_keys != expected_keys:
        errors.append(f"Explanation root keys must be exactly {expected_keys}, received {actual_keys}.")
    missing = [key for key in required if key not in payload]
    extra = [key for key in actual_keys if key not in properties]
    if missing:
        errors.append(f"Explanation payload missing required keys: {', '.join(missing)}.")
    if extra:
        errors.append(f"Explanation payload contains unexpected keys: {', '.join(extra)}.")
    errors.extend(validate_explanation_text(payload.get("explanation")))
    return errors


def validate_sparse_schema_node(
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

    if path == "" and key_order_root_expected is not None and actual_keys != key_order_root_expected:
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
            child_schema_errors, child_range_errors, child_key_order = validate_sparse_schema_node(
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
                range_errors.append(
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
                range_errors.append(f"{child_path} is below minimum {minimum}: received {numeric_value}.")
            if maximum is not None and numeric_value > float(maximum):
                range_errors.append(f"{child_path} is above maximum {maximum}: received {numeric_value}.")
            continue

        schema_errors.append(
            f"{child_path} has incompatible type {type(child_value).__name__}; allowed types are {sorted(type_set)}."
        )

    return schema_errors, range_errors, key_order_pass


def validate_sparse_patch_payload(
    payload: dict[str, Any],
    schema: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[list[str], list[str], bool, dict[str, Any], list[str]]:
    schema_errors, range_errors, key_order_pass = validate_sparse_schema_node(
        schema,
        payload,
        path="",
        key_order_root_expected=PATCH_BLOCK_ORDER,
    )
    changed_paths: dict[str, Any] = {}
    ignored_paths: list[str] = []
    if not schema_errors:
        for path in registry["parameter_order"]:
            parts = path.split(".")
            cursor: Any = payload
            for part in parts:
                if not isinstance(cursor, dict) or part not in cursor:
                    cursor = None
                    break
                cursor = cursor[part]
            if cursor is None:
                ignored_paths.append(path)
            else:
                changed_paths[path] = cursor
    return schema_errors, range_errors, key_order_pass, changed_paths, ignored_paths


def validate_compact_delta_payload(
    payload: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any], list[str]]:
    schema_errors: list[str] = []
    range_errors: list[str] = []
    semantic_errors: list[str] = []
    changed_paths: dict[str, Any] = {}

    actual_keys = list(payload.keys())
    if actual_keys != ["changes"]:
        schema_errors.append(f"Root keys must be exactly ['changes'], received {actual_keys}.")

    changes = payload.get("changes")
    if not isinstance(changes, list):
        schema_errors.append("changes must be an array.")
        return schema_errors, range_errors, changed_paths, semantic_errors

    seen_paths: set[str] = set()
    for index, item in enumerate(changes):
        item_path = f"changes[{index}]"
        if not isinstance(item, dict):
            schema_errors.append(f"{item_path} must be an object, received {type(item).__name__}.")
            continue

        item_keys = list(item.keys())
        if item_keys != ["path", "value"]:
            schema_errors.append(
                f"{item_path} keys must be exactly ['path', 'value'], received {item_keys}."
            )

        path_value = item.get("path")
        change_value = item.get("value")
        if not isinstance(path_value, str):
            schema_errors.append(f"{item_path}.path must be a string.")
            continue
        if path_value not in registry["parameters"]:
            schema_errors.append(f"{item_path}.path is not a supported leaf path: {path_value!r}.")
            continue
        if path_value in seen_paths:
            semantic_errors.append(f"{item_path}.path is duplicated: {path_value!r}.")
            continue

        spec = registry["parameters"][path_value]
        seen_paths.add(path_value)

        value_type = spec["value_type"]
        if value_type == "enum-string":
            if not isinstance(change_value, str):
                semantic_errors.append(
                    f"{item_path}.value for {path_value} must be a string, received {type(change_value).__name__}."
                )
                continue
            if change_value not in spec["allowed"]:
                semantic_errors.append(
                    f"{item_path}.value for {path_value} must be one of {spec['allowed']}, received {change_value!r}."
                )
                continue
            changed_paths[path_value] = change_value
            continue

        if not isinstance(change_value, (int, float)) or isinstance(change_value, bool):
            semantic_errors.append(
                f"{item_path}.value for {path_value} must be numeric, received {type(change_value).__name__}."
            )
            continue
        numeric_value = float(change_value)
        if not math.isfinite(numeric_value):
            semantic_errors.append(f"{item_path}.value for {path_value} must be finite, received {change_value!r}.")
            continue

        minimum = spec.get("min")
        maximum = spec.get("max")
        if minimum is not None and numeric_value < float(minimum):
            range_errors.append(f"{path_value} is below minimum {minimum}: received {numeric_value}.")
        if maximum is not None and numeric_value > float(maximum):
            range_errors.append(f"{path_value} is above maximum {maximum}: received {numeric_value}.")
        if value_type == "enum-int" and numeric_value not in [float(value) for value in spec["allowed"]]:
            range_errors.append(
                f"{path_value} must be one of {spec['allowed']}, received {numeric_value}."
            )
        changed_paths[path_value] = change_value

    return schema_errors, range_errors, changed_paths, semantic_errors


def compute_changed_path_metrics(
    *,
    changed_paths: dict[str, Any],
    ignored_paths: list[str],
    registry: dict[str, Any],
    patch_contract: str,
) -> dict[str, Any]:
    changed_blocks = sorted({path.split(".", 1)[0] for path in changed_paths})
    total_leaf_count = len(registry["parameter_order"])
    if patch_contract == PATCH_CONTRACT_SPARSE:
        null_param_count = len(ignored_paths)
        omitted_param_count = 0
    else:
        null_param_count = 0
        omitted_param_count = total_leaf_count - len(changed_paths)
    null_ratio = (null_param_count / total_leaf_count) if total_leaf_count else 0.0
    omitted_ratio = (omitted_param_count / total_leaf_count) if total_leaf_count else 0.0
    return {
        "changed_param_count": len(changed_paths),
        "changed_block_count": len(changed_blocks),
        "changed_blocks": changed_blocks,
        "null_param_count": null_param_count,
        "null_ratio": null_ratio,
        "omitted_param_count": omitted_param_count,
        "omitted_ratio": omitted_ratio,
        "all_null_noop": len(changed_paths) == 0,
    }


def values_equal_for_path(path: str, left: Any, right: Any, registry: dict[str, Any]) -> bool:
    spec = registry["parameters"][path]
    if left is None or right is None:
        return left is right
    if spec["value_type"] in {"enum-string", "enum-int"}:
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        epsilon = float(spec.get("compare_epsilon", 0.0))
        return abs(float(left) - float(right)) <= epsilon
    return left == right


def evaluate_sparse_fidelity(
    *,
    prompt_case: PromptCase,
    changed_paths: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    unchanged_non_null_paths: list[str] = []
    current_paths = (
        flatten_leaf_paths(prompt_case.current_patch_context)
        if prompt_case.current_patch_context is not None
        else {}
    )
    for path, value in changed_paths.items():
        if path in current_paths and values_equal_for_path(path, value, current_paths[path], registry):
            unchanged_non_null_paths.append(path)

    expected_missing_paths: list[str] = []
    unexpected_changed_paths: list[str] = []
    mismatched_value_paths: list[str] = []
    sparse_exact_match: bool | None = None
    expected_changed_path_set: set[str] | None = None
    if prompt_case.expected_changed_paths is not None or prompt_case.expected_exact_values is not None:
        expected_changed_path_set = set(prompt_case.expected_changed_paths or [])
        if prompt_case.expected_exact_values is not None:
            expected_changed_path_set.update(prompt_case.expected_exact_values.keys())
        sparse_exact_match = True
        for expected_path in sorted(expected_changed_path_set):
            if expected_path not in changed_paths:
                expected_missing_paths.append(expected_path)
                sparse_exact_match = False
        if prompt_case.expected_exact_values is not None:
            for expected_path, expected_value in prompt_case.expected_exact_values.items():
                if expected_path not in changed_paths:
                    continue
                if not values_equal_for_path(expected_path, changed_paths[expected_path], expected_value, registry):
                    mismatched_value_paths.append(expected_path)
                    sparse_exact_match = False
        for actual_path in changed_paths:
            if actual_path not in expected_changed_path_set:
                unexpected_changed_paths.append(actual_path)
                sparse_exact_match = False

    if prompt_case.expected_null_paths is not None:
        for path in prompt_case.expected_null_paths:
            if path in changed_paths:
                unexpected_changed_paths.append(path)
                if sparse_exact_match is None:
                    sparse_exact_match = False
                else:
                    sparse_exact_match = False

    noop_exact_pass: bool | None = None
    if prompt_case.expected_noop is not None:
        noop_exact_pass = (len(changed_paths) == 0) if prompt_case.expected_noop else (len(changed_paths) > 0)

    return {
        "unchanged_non_null_paths": sorted(set(unchanged_non_null_paths)),
        "false_edit_count": len(set(unchanged_non_null_paths)),
        "expected_missing_paths": sorted(set(expected_missing_paths)),
        "unexpected_changed_paths": sorted(set(unexpected_changed_paths)),
        "mismatched_value_paths": sorted(set(mismatched_value_paths)),
        "sparse_exact_match": sparse_exact_match,
        "noop_exact_pass": noop_exact_pass,
    }


def normalize_patch_payload(
    *,
    payload: dict[str, Any],
    patch_contract: str,
    patch_schema: dict[str, Any],
    registry: dict[str, Any],
    prompt_case: PromptCase,
) -> dict[str, Any]:
    if patch_contract == PATCH_CONTRACT_SPARSE:
        schema_errors, range_errors, key_order_pass, changed_paths, ignored_paths = validate_sparse_patch_payload(
            payload,
            patch_schema,
            registry,
        )
        semantic_errors: list[str] = []
        key_order_errors = [] if key_order_pass else ["Sparse patch root key order failed validation."]
    else:
        schema_errors, range_errors, changed_paths, semantic_errors = validate_compact_delta_payload(
            payload,
            registry,
        )
        ignored_paths = [path for path in registry["parameter_order"] if path not in changed_paths]
        key_order_pass = True
        key_order_errors = []

    metrics = compute_changed_path_metrics(
        changed_paths=changed_paths,
        ignored_paths=ignored_paths,
        registry=registry,
        patch_contract=patch_contract,
    )
    sparse_fidelity = evaluate_sparse_fidelity(
        prompt_case=prompt_case,
        changed_paths=changed_paths,
        registry=registry,
    )
    return {
        "schema_errors": schema_errors,
        "range_errors": range_errors,
        "semantic_errors": semantic_errors,
        "key_order_pass": key_order_pass,
        "key_order_errors": key_order_errors,
        "changed_paths": changed_paths,
        "ignored_paths": ignored_paths,
        "metrics": metrics,
        "sparse_fidelity": sparse_fidelity,
    }


def sanitize_leaf_value(
    *,
    path: str,
    raw_value: Any,
    spec: dict[str, Any],
    current_value_before_apply: Any,
) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    value_type = spec["value_type"]
    policy = spec["sanitize_policy"]

    if raw_value is None:
        return None, None, None

    if value_type == "enum-string":
        if not isinstance(raw_value, str):
            return None, None, f"{path} expects a string enum, received {type(raw_value).__name__}."
        if raw_value not in spec["allowed"]:
            return None, None, f"{path} must be one of {spec['allowed']}, received {raw_value!r}."
        return raw_value, None, None

    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        return None, None, f"{path} expects a numeric value, received {type(raw_value).__name__}."

    numeric_value = float(raw_value)
    if not math.isfinite(numeric_value):
        return None, None, f"{path} must be finite, received {raw_value!r}."

    sanitized_value: Any = raw_value
    action = None
    reason = None

    if policy == "round_then_clamp":
        rounded_value = int(round(numeric_value))
        if rounded_value != raw_value:
            action = "round_then_clamp"
            reason = "integer_rounding"
        sanitized_value = rounded_value
        minimum = int(spec["min"])
        maximum = int(spec["max"])
        clamped_value = min(max(int(sanitized_value), minimum), maximum)
        if clamped_value != sanitized_value:
            action = "round_then_clamp"
            reason = "integer_range_clamp"
        sanitized_value = clamped_value
    elif policy == "hard_clamp":
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        clamped_value = min(max(float(numeric_value), minimum), maximum)
        if not math.isclose(clamped_value, numeric_value, rel_tol=0.0, abs_tol=0.0):
            action = "hard_clamp"
            reason = "numeric_range_clamp"
        sanitized_value = clamped_value
    elif policy == "nearest_allowed":
        allowed = spec["allowed"]
        sanitized_value = min(allowed, key=lambda candidate: abs(float(candidate) - numeric_value))
        if float(sanitized_value) != numeric_value:
            action = "nearest_allowed"
            reason = "discrete_numeric_snap"
    elif policy == "reject":
        return None, None, f"{path} uses reject policy and cannot coerce raw value {raw_value!r}."
    else:
        return None, None, f"{path} has unsupported sanitize policy {policy!r}."

    if value_type == "int":
        sanitized_value = int(sanitized_value)
    elif value_type == "enum-int":
        sanitized_value = int(sanitized_value)
    else:
        sanitized_value = float(sanitized_value)

    clamp_event = None
    if action is not None:
        absolute_delta = None
        relative_delta = None
        if isinstance(raw_value, (int, float)):
            absolute_delta = abs(float(raw_value) - float(sanitized_value))
            if float(raw_value) != 0.0:
                relative_delta = absolute_delta / abs(float(raw_value))
        clamp_event = {
            "path": path,
            "raw_value": raw_value,
            "sanitized_value": sanitized_value,
            "action": action,
            "reason": reason,
            "min": spec.get("min"),
            "max": spec.get("max"),
            "allowed": spec.get("allowed"),
            "absolute_delta": absolute_delta,
            "relative_delta": relative_delta,
            "current_value_before_apply": current_value_before_apply,
            "applied_value_after_merge": sanitized_value,
            "collapsed_to_noop": False,
        }

    return sanitized_value, clamp_event, None


def merge_changed_paths_into_context(
    current_patch_context: dict[str, Any] | None,
    changed_paths: dict[str, Any],
) -> dict[str, Any] | None:
    if current_patch_context is None:
        return None
    merged = deep_copy_json(current_patch_context)
    for path, value in changed_paths.items():
        assign_nested_path(merged, path, value)
    return merged


def apply_sanitized_changes(
    *,
    prompt_case: PromptCase,
    raw_changed_paths: dict[str, Any],
    registry: dict[str, Any],
    patch_contract: str,
) -> dict[str, Any]:
    current_paths = (
        flatten_leaf_paths(prompt_case.current_patch_context)
        if prompt_case.current_patch_context is not None
        else {}
    )
    sanitized_changed_paths: dict[str, Any] = {}
    apply_noop_paths: list[str] = []
    clamp_events: list[dict[str, Any]] = []
    apply_errors: list[str] = []

    for path in registry["parameter_order"]:
        if path not in raw_changed_paths:
            continue
        raw_value = raw_changed_paths[path]
        spec = registry["parameters"][path]
        current_value_before_apply = current_paths.get(path)
        sanitized_value, clamp_event, error = sanitize_leaf_value(
            path=path,
            raw_value=raw_value,
            spec=spec,
            current_value_before_apply=current_value_before_apply,
        )
        if error is not None:
            apply_errors.append(error)
            continue
        if clamp_event is not None:
            clamp_events.append(clamp_event)

        if path in current_paths and values_equal_for_path(path, sanitized_value, current_paths[path], registry):
            apply_noop_paths.append(path)
            if clamp_event is not None:
                clamp_event["collapsed_to_noop"] = True
            continue

        sanitized_changed_paths[path] = sanitized_value

    merged_patch_preview = merge_changed_paths_into_context(
        prompt_case.current_patch_context,
        sanitized_changed_paths,
    )
    sanitized_sparse_patch_preview = build_sparse_patch_preview(sanitized_changed_paths, registry)
    sanitized_compact_delta_preview = build_compact_delta_preview(sanitized_changed_paths, registry)
    ignored_paths = [path for path in registry["parameter_order"] if path not in sanitized_changed_paths]
    metrics = compute_changed_path_metrics(
        changed_paths=sanitized_changed_paths,
        ignored_paths=ignored_paths,
        registry=registry,
        patch_contract=patch_contract,
    )
    sparse_fidelity = evaluate_sparse_fidelity(
        prompt_case=prompt_case,
        changed_paths=sanitized_changed_paths,
        registry=registry,
    )

    return {
        "apply_errors": apply_errors,
        "apply_noop_paths": apply_noop_paths,
        "clamp_events": clamp_events,
        "clamped_paths": sorted({event["path"] for event in clamp_events}),
        "sanitized_changed_paths": sanitized_changed_paths,
        "sanitized_sparse_patch_preview": sanitized_sparse_patch_preview,
        "sanitized_compact_delta_preview": sanitized_compact_delta_preview,
        "merged_patch_preview": merged_patch_preview,
        "metrics": metrics,
        "sparse_fidelity": sparse_fidelity,
    }


def build_parameter_description_lines(registry: dict[str, Any], *, include_null: bool) -> list[str]:
    lines: list[str] = []
    for path in registry["parameter_order"]:
        spec = registry["parameters"][path]
        value_type = spec["value_type"]
        if value_type == "enum-string":
            allowed_text = ", ".join(repr(value) for value in spec["allowed"])
            suffix = ", or null" if include_null else ""
            lines.append(f"- {path}: one of {allowed_text}{suffix}")
        elif value_type == "enum-int":
            allowed_text = ", ".join(str(value) for value in spec["allowed"])
            suffix = ", or null" if include_null else ""
            lines.append(f"- {path}: one of {allowed_text}{suffix}")
        else:
            min_value = spec["min"]
            max_value = spec["max"]
            numeric_kind = "integer" if value_type == "int" else "number"
            suffix = ", or null" if include_null else ""
            lines.append(f"- {path}: {numeric_kind} {min_value} to {max_value}{suffix}")
    return lines


def default_patch_system_prompt(patch_contract: str, registry: dict[str, Any]) -> str:
    parameter_lines = build_parameter_description_lines(
        registry,
        include_null=(patch_contract == PATCH_CONTRACT_SPARSE),
    )
    if patch_contract == PATCH_CONTRACT_SPARSE:
        header = [
            "You are the Sound Forge Experiment 1.2 patch engine.",
            "Return exactly one JSON object and nothing else.",
            "Do not add markdown, code fences, comments, or explanation text.",
            "Emit these exact top-level blocks in this exact order: oscillator-1, oscillator-2, filter, envelope, effects.",
            "This is a sparse patch contract. Use null aggressively for unchanged parameters.",
            "If current patch context is provided, output non-null values only for real changes relative to that context.",
            "Do not copy unchanged values into non-null fields.",
            "If nothing should change, return all parameters as null.",
            "Supported parameters:",
        ]
    else:
        header = [
            "You are the Sound Forge Experiment 1.2 compact delta patch engine.",
            "Return exactly one JSON object and nothing else.",
            "Do not add markdown, code fences, comments, or explanation text.",
            "The JSON object must have exactly one key named changes.",
            "changes must be an array of objects with keys path and value in that order.",
            "Do not emit null values.",
            "Do not repeat the same path more than once.",
            "If current patch context is provided, omit paths that already equal the current value.",
            "If nothing should change, return {\"changes\": []}.",
            "Supported paths and value families:",
        ]
    return "\n".join([*header, *parameter_lines])


def default_explanation_system_prompt() -> str:
    return "\n".join(
        [
            "You are the Sound Forge explanation worker.",
            "You receive a user request and the final sanitized applied patch after validation and clamping.",
            "Describe only the final sanitized patch, never the rejected raw model output.",
            "Return exactly one JSON object with one key named explanation.",
            "Keep the explanation between 20 and 280 characters.",
            "Keep it practical and UI-facing.",
        ]
    )


def load_patch_system_prompt(
    path: Path | None,
    patch_contract: str,
    registry: dict[str, Any],
) -> str:
    if path is not None:
        ensure_file(path, "Patch system prompt file")
        return load_text(path)
    return default_patch_system_prompt(patch_contract, registry)


def load_explanation_system_prompt(path: Path | None) -> str:
    if path is not None:
        ensure_file(path, "Explanation system prompt file")
        return load_text(path)
    return default_explanation_system_prompt()


def build_patch_messages(
    *,
    system_prompt: str,
    prompt_case: PromptCase,
    few_shot_examples: list[FewShotExample],
) -> list[dict[str, str]]:
    user_parts: list[str] = []
    if prompt_case.current_patch_context is not None:
        user_parts.append("Current patch context (JSON):")
        user_parts.append(json.dumps(prompt_case.current_patch_context, indent=2))
        user_parts.append("")
    user_parts.append("User request:")
    user_parts.append(prompt_case.prompt)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for example in few_shot_examples:
        messages.append({"role": "user", "content": example.user})
        messages.append({"role": "assistant", "content": example.assistant})
    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages


def build_explanation_messages(
    *,
    system_prompt: str,
    prompt_case: PromptCase,
    sanitized_changed_paths: dict[str, Any],
    sanitized_sparse_patch_preview: dict[str, Any],
    merged_patch_preview: dict[str, Any] | None,
) -> list[dict[str, str]]:
    user_parts = [
        "User request:",
        prompt_case.prompt,
        "",
        "Final sanitized applied changed paths (JSON):",
        json.dumps(sanitized_changed_paths, indent=2),
        "",
        "Final sanitized sparse patch preview (JSON):",
        json.dumps(sanitized_sparse_patch_preview, indent=2),
    ]
    if merged_patch_preview is not None:
        user_parts.extend(
            [
                "",
                "Merged patch preview after apply (JSON):",
                json.dumps(merged_patch_preview, indent=2),
            ]
        )
    user_parts.extend(
        [
            "",
            "Only describe the sanitized final patch above.",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def invoke_chat_completion(
    *,
    handle: ModelHandle,
    messages: list[dict[str, str]],
    grammar: LlamaGrammar,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    max_tokens: int,
) -> dict[str, Any]:
    with handle.lock:
        return handle.llm.create_chat_completion(
            messages=messages,
            grammar=grammar,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
        )


def initialize_record(
    *,
    prompt_case: PromptCase,
    patch_contract: str,
    explanation_mode: str,
    submitted_index: int,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_case.case_id,
        "stream_id": prompt_case.stream_id,
        "submitted_index": submitted_index,
        "prompt": prompt_case.prompt,
        "tags": prompt_case.tags,
        "expected_changed_paths": prompt_case.expected_changed_paths,
        "expected_exact_values": prompt_case.expected_exact_values,
        "expected_null_paths": prompt_case.expected_null_paths,
        "expected_clamped_paths": prompt_case.expected_clamped_paths,
        "expected_noop": prompt_case.expected_noop,
        "patch_contract": patch_contract,
        "explanation_mode": explanation_mode,
        "current_patch_context": prompt_case.current_patch_context,
        "patch_queue_wait_ms": None,
        "apply_queue_wait_ms": None,
        "explanation_queue_wait_ms": None,
        "patch_latency_ms": None,
        "patch_parse_latency_ms": None,
        "patch_validation_latency_ms": None,
        "apply_latency_ms": None,
        "explanation_latency_ms": None,
        "patch_ready_ms": None,
        "patch_applied_ms": None,
        "explanation_ready_ms": None,
        "ui_gap_ms": None,
        "patch_raw_output": "",
        "patch_trimmed_output": "",
        "patch_parsed_payload": None,
        "patch_generation_error": None,
        "patch_finish_reason": None,
        "patch_prompt_tokens": None,
        "patch_completion_tokens": None,
        "patch_total_tokens": None,
        "patch_tokens_per_second": None,
        "raw_syntax_pass": False,
        "raw_schema_pass": False,
        "raw_key_order_pass": False,
        "raw_range_pass": False,
        "raw_semantic_pass": False,
        "raw_validation_errors": [],
        "raw_changed_paths": {},
        "raw_ignored_paths": [],
        "raw_sparse_patch_preview": None,
        "raw_compact_delta_preview": None,
        "raw_changed_param_count": 0,
        "raw_changed_block_count": 0,
        "raw_changed_blocks": [],
        "raw_null_param_count": 0,
        "raw_null_ratio": None,
        "raw_omitted_param_count": 0,
        "raw_omitted_ratio": None,
        "raw_all_null_noop": False,
        "raw_sparse_fidelity": {
            "unchanged_non_null_paths": [],
            "false_edit_count": 0,
            "expected_missing_paths": [],
            "unexpected_changed_paths": [],
            "mismatched_value_paths": [],
            "sparse_exact_match": None,
            "noop_exact_pass": None,
        },
        "apply_errors": [],
        "apply_success": False,
        "post_clamp_apply_pass": False,
        "apply_noop_paths": [],
        "clamp_events": [],
        "clamp_event_count": 0,
        "clamp_intervention": False,
        "clamped_paths": [],
        "clamp_expectation_pass": None,
        "missing_expected_clamped_paths": [],
        "unexpected_clamped_paths": [],
        "sanitized_changed_paths": {},
        "sanitized_sparse_patch_preview": None,
        "sanitized_compact_delta_preview": None,
        "merged_patch_preview": None,
        "sanitized_changed_param_count": 0,
        "sanitized_changed_block_count": 0,
        "sanitized_changed_blocks": [],
        "sanitized_null_param_count": 0,
        "sanitized_null_ratio": None,
        "sanitized_omitted_param_count": 0,
        "sanitized_omitted_ratio": None,
        "sanitized_all_null_noop": False,
        "sanitized_sparse_fidelity": {
            "unchanged_non_null_paths": [],
            "false_edit_count": 0,
            "expected_missing_paths": [],
            "unexpected_changed_paths": [],
            "mismatched_value_paths": [],
            "sparse_exact_match": None,
            "noop_exact_pass": None,
        },
        "patch_pipeline_pass": False,
        "explanation_requested": explanation_mode != EXPLANATION_MODE_OFF,
        "explanation_available": False,
        "explanation_pass": False,
        "explanation_cancelled": False,
        "explanation_discarded": False,
        "explanation_errors": [],
        "explanation_generation_error": None,
        "explanation_finish_reason": None,
        "explanation_prompt_tokens": None,
        "explanation_completion_tokens": None,
        "explanation_total_tokens": None,
        "explanation_tokens_per_second": None,
        "explanation_raw_output": "",
        "explanation_trimmed_output": "",
        "explanation_payload": None,
        "explanation_text": None,
    }


def patch_worker_loop(
    *,
    reporter: StatusReporter,
    input_queue: queue.Queue[PipelineJob | object],
    apply_queue: queue.Queue[PipelineJob | object],
    completion_queue: queue.Queue[PipelineJob | object],
    shared_state: SharedPipelineState,
    patch_handle: ModelHandle,
    patch_grammar: LlamaGrammar,
    patch_schema: dict[str, Any],
    patch_system_prompt: str,
    patch_contract: str,
    registry: dict[str, Any],
    few_shot_examples: list[FewShotExample],
    args: argparse.Namespace,
) -> None:
    while True:
        item = input_queue.get()
        if item is SENTINEL:
            apply_queue.put(SENTINEL)
            shared_state.record_queue_size("apply", apply_queue.qsize())
            return

        assert isinstance(item, PipelineJob)
        prompt_case = item.prompt_case
        record = item.record
        patch_started = perf_counter()
        record["patch_queue_wait_ms"] = (patch_started - item.patch_queue_entered_perf) * 1000.0

        messages = build_patch_messages(
            system_prompt=patch_system_prompt,
            prompt_case=prompt_case,
            few_shot_examples=few_shot_examples,
        )

        patch_generation_started = perf_counter()
        response: dict[str, Any] | None = None
        raw_output = ""
        generation_error: str | None = None
        try:
            response = invoke_chat_completion(
                handle=patch_handle,
                messages=messages,
                grammar=patch_grammar,
                temperature=args.patch_temperature,
                top_p=args.patch_top_p,
                repeat_penalty=args.patch_repeat_penalty,
                max_tokens=args.patch_max_tokens,
            )
            choice = response["choices"][0]
            raw_output = normalize_assistant_content(choice["message"].get("content"))
        except Exception as exc:  # pragma: no cover - runtime dependent
            generation_error = str(exc)
        patch_generation_ended = perf_counter()

        record["patch_latency_ms"] = (patch_generation_ended - patch_generation_started) * 1000.0
        record["patch_raw_output"] = raw_output
        record["patch_trimmed_output"] = raw_output.strip()
        record["patch_generation_error"] = generation_error

        if response is not None:
            choice = response["choices"][0]
            record["patch_finish_reason"] = choice.get("finish_reason")
            usage = response.get("usage", {})
            record["patch_prompt_tokens"] = usage.get("prompt_tokens")
            record["patch_completion_tokens"] = usage.get("completion_tokens")
            record["patch_total_tokens"] = usage.get("total_tokens")
            if record["patch_completion_tokens"] and record["patch_latency_ms"] and record["patch_latency_ms"] > 0:
                record["patch_tokens_per_second"] = record["patch_completion_tokens"] / (
                    record["patch_latency_ms"] / 1000.0
                )

        if generation_error is not None:
            record["raw_validation_errors"] = [f"Patch generation error: {generation_error}"]
            record["patch_ready_ms"] = (patch_generation_ended - item.submitted_perf) * 1000.0
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        parse_started = perf_counter()
        parsed_payload, syntax_errors = parse_json_object(raw_output)
        parse_ended = perf_counter()
        record["patch_parse_latency_ms"] = (parse_ended - parse_started) * 1000.0
        if syntax_errors:
            record["raw_validation_errors"] = syntax_errors
            record["patch_ready_ms"] = (parse_ended - item.submitted_perf) * 1000.0
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        record["raw_syntax_pass"] = True
        record["patch_parsed_payload"] = parsed_payload

        validation_started = perf_counter()
        normalized = normalize_patch_payload(
            payload=parsed_payload,
            patch_contract=patch_contract,
            patch_schema=patch_schema,
            registry=registry,
            prompt_case=prompt_case,
        )
        validation_ended = perf_counter()
        record["patch_validation_latency_ms"] = (validation_ended - validation_started) * 1000.0
        record["patch_ready_ms"] = (validation_ended - item.submitted_perf) * 1000.0

        record["raw_schema_pass"] = not normalized["schema_errors"]
        record["raw_key_order_pass"] = normalized["key_order_pass"]
        record["raw_range_pass"] = not normalized["range_errors"]
        record["raw_semantic_pass"] = not normalized["semantic_errors"]
        record["raw_validation_errors"] = [
            *normalized["schema_errors"],
            *normalized["range_errors"],
            *normalized["semantic_errors"],
            *normalized["key_order_errors"],
        ]

        raw_changed_paths = normalized["changed_paths"]
        record["raw_changed_paths"] = raw_changed_paths
        record["raw_ignored_paths"] = normalized["ignored_paths"]
        record["raw_sparse_patch_preview"] = build_sparse_patch_preview(raw_changed_paths, registry)
        record["raw_compact_delta_preview"] = build_compact_delta_preview(raw_changed_paths, registry)
        record["raw_changed_param_count"] = normalized["metrics"]["changed_param_count"]
        record["raw_changed_block_count"] = normalized["metrics"]["changed_block_count"]
        record["raw_changed_blocks"] = normalized["metrics"]["changed_blocks"]
        record["raw_null_param_count"] = normalized["metrics"]["null_param_count"]
        record["raw_null_ratio"] = normalized["metrics"]["null_ratio"]
        record["raw_omitted_param_count"] = normalized["metrics"]["omitted_param_count"]
        record["raw_omitted_ratio"] = normalized["metrics"]["omitted_ratio"]
        record["raw_all_null_noop"] = normalized["metrics"]["all_null_noop"]
        record["raw_sparse_fidelity"] = normalized["sparse_fidelity"]

        hard_fail = (
            not record["raw_schema_pass"]
            or not record["raw_semantic_pass"]
            or not record["raw_key_order_pass"]
        )
        if hard_fail:
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        item.apply_queue_entered_perf = perf_counter()
        apply_queue.put(item)
        shared_state.record_queue_size("apply", apply_queue.qsize())


def apply_worker_loop(
    *,
    reporter: StatusReporter,
    apply_queue: queue.Queue[PipelineJob | object],
    explanation_queue: queue.Queue[PipelineJob | object] | None,
    completion_queue: queue.Queue[PipelineJob | object],
    shared_state: SharedPipelineState,
    registry: dict[str, Any],
    explanation_mode: str,
) -> None:
    while True:
        item = apply_queue.get()
        if item is SENTINEL:
            if explanation_queue is not None:
                explanation_queue.put(SENTINEL)
                shared_state.record_queue_size("explanation", explanation_queue.qsize())
            else:
                completion_queue.put(SENTINEL)
                shared_state.record_queue_size("completion", completion_queue.qsize())
            return

        assert isinstance(item, PipelineJob)
        apply_started = perf_counter()
        if item.apply_queue_entered_perf is not None:
            item.record["apply_queue_wait_ms"] = (apply_started - item.apply_queue_entered_perf) * 1000.0

        apply_compute_started = perf_counter()
        apply_result = apply_sanitized_changes(
            prompt_case=item.prompt_case,
            raw_changed_paths=item.record["raw_changed_paths"],
            registry=registry,
            patch_contract=item.record["patch_contract"],
        )
        apply_compute_ended = perf_counter()

        record = item.record
        record["apply_latency_ms"] = (apply_compute_ended - apply_compute_started) * 1000.0
        record["patch_applied_ms"] = (apply_compute_ended - item.submitted_perf) * 1000.0
        record["apply_errors"] = apply_result["apply_errors"]
        record["apply_noop_paths"] = apply_result["apply_noop_paths"]
        record["clamp_events"] = apply_result["clamp_events"]
        record["clamp_event_count"] = len(apply_result["clamp_events"])
        record["clamp_intervention"] = bool(apply_result["clamp_events"])
        record["clamped_paths"] = apply_result["clamped_paths"]
        if item.prompt_case.expected_clamped_paths is not None:
            expected_clamped = set(item.prompt_case.expected_clamped_paths)
            actual_clamped = set(record["clamped_paths"])
            record["missing_expected_clamped_paths"] = sorted(expected_clamped - actual_clamped)
            record["unexpected_clamped_paths"] = sorted(actual_clamped - expected_clamped)
            record["clamp_expectation_pass"] = (
                not record["missing_expected_clamped_paths"]
                and not record["unexpected_clamped_paths"]
            )
        record["sanitized_changed_paths"] = apply_result["sanitized_changed_paths"]
        record["sanitized_sparse_patch_preview"] = apply_result["sanitized_sparse_patch_preview"]
        record["sanitized_compact_delta_preview"] = apply_result["sanitized_compact_delta_preview"]
        record["merged_patch_preview"] = apply_result["merged_patch_preview"]
        record["sanitized_changed_param_count"] = apply_result["metrics"]["changed_param_count"]
        record["sanitized_changed_block_count"] = apply_result["metrics"]["changed_block_count"]
        record["sanitized_changed_blocks"] = apply_result["metrics"]["changed_blocks"]
        record["sanitized_null_param_count"] = apply_result["metrics"]["null_param_count"]
        record["sanitized_null_ratio"] = apply_result["metrics"]["null_ratio"]
        record["sanitized_omitted_param_count"] = apply_result["metrics"]["omitted_param_count"]
        record["sanitized_omitted_ratio"] = apply_result["metrics"]["omitted_ratio"]
        record["sanitized_all_null_noop"] = apply_result["metrics"]["all_null_noop"]
        record["sanitized_sparse_fidelity"] = apply_result["sparse_fidelity"]
        record["apply_success"] = not record["apply_errors"]
        record["post_clamp_apply_pass"] = record["apply_success"]
        record["patch_pipeline_pass"] = (
            record["raw_syntax_pass"]
            and record["raw_schema_pass"]
            and record["raw_semantic_pass"]
            and record["raw_key_order_pass"]
            and record["post_clamp_apply_pass"]
        )

        if not record["apply_success"] or explanation_mode == EXPLANATION_MODE_OFF:
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        item.patch_version = shared_state.next_patch_version(item.prompt_case.stream_id)
        item.explanation_queue_entered_perf = perf_counter()
        assert explanation_queue is not None
        explanation_queue.put(item)
        shared_state.record_queue_size("explanation", explanation_queue.qsize())


def deterministic_explanation_from_record(record: dict[str, Any]) -> str:
    changed_paths = record["sanitized_changed_paths"]
    if not changed_paths:
        return "The final sanitized patch leaves all synth parameters unchanged because no safe edits were needed."

    fragments: list[str] = []
    for index, (path, value) in enumerate(changed_paths.items()):
        if index >= 4:
            break
        fragments.append(f"{path}={value}")

    prefix = "The final sanitized patch applies "
    if len(changed_paths) == 1:
        body = fragments[0]
    else:
        body = ", ".join(fragments[:-1]) + f", and {fragments[-1]}" if len(fragments) > 1 else fragments[0]
    suffix = "."
    if len(changed_paths) > 4:
        suffix = f", plus {len(changed_paths) - 4} additional safe parameter updates."
    explanation = prefix + body + suffix
    if len(explanation) > 280:
        explanation = explanation[:277].rstrip() + "..."
    return explanation


def explanation_worker_loop(
    *,
    reporter: StatusReporter,
    explanation_queue: queue.Queue[PipelineJob | object],
    completion_queue: queue.Queue[PipelineJob | object],
    shared_state: SharedPipelineState,
    explanation_mode: str,
    explanation_handle: ModelHandle | None,
    explanation_grammar: LlamaGrammar | None,
    explanation_schema: dict[str, Any] | None,
    explanation_system_prompt: str,
    args: argparse.Namespace,
) -> None:
    while True:
        item = explanation_queue.get()
        if item is SENTINEL:
            completion_queue.put(SENTINEL)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            return

        assert isinstance(item, PipelineJob)
        assert item.patch_version is not None
        record = item.record
        explanation_started = perf_counter()
        if item.explanation_queue_entered_perf is not None:
            record["explanation_queue_wait_ms"] = (
                explanation_started - item.explanation_queue_entered_perf
            ) * 1000.0

        latest_patch_version = shared_state.latest_patch_version(item.prompt_case.stream_id)
        if item.patch_version < latest_patch_version:
            record["explanation_cancelled"] = True
            record["explanation_errors"] = [
                f"Explanation cancelled because stream {item.prompt_case.stream_id!r} advanced to patch version {latest_patch_version}."
            ]
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        if explanation_mode == EXPLANATION_MODE_DETERMINISTIC:
            generation_started = perf_counter()
            explanation_text = deterministic_explanation_from_record(record)
            generation_ended = perf_counter()
            latest_patch_version = shared_state.latest_patch_version(item.prompt_case.stream_id)
            if item.patch_version < latest_patch_version:
                record["explanation_discarded"] = True
                record["explanation_errors"] = [
                    f"Explanation discarded because stream {item.prompt_case.stream_id!r} advanced to patch version {latest_patch_version}."
                ]
                completion_queue.put(item)
                shared_state.record_queue_size("completion", completion_queue.qsize())
                continue
            record["explanation_latency_ms"] = (generation_ended - generation_started) * 1000.0
            record["explanation_ready_ms"] = (generation_ended - item.submitted_perf) * 1000.0
            if record["patch_applied_ms"] is not None:
                record["ui_gap_ms"] = record["explanation_ready_ms"] - record["patch_applied_ms"]
            record["explanation_text"] = explanation_text
            record["explanation_payload"] = {"explanation": explanation_text}
            deterministic_errors = validate_explanation_payload(
                record["explanation_payload"],
                explanation_schema or {"properties": {"explanation": {}}, "required": ["explanation"]},
            )
            record["explanation_available"] = not deterministic_errors
            record["explanation_pass"] = not deterministic_errors
            if deterministic_errors:
                record["explanation_errors"] = deterministic_errors
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        assert explanation_mode == EXPLANATION_MODE_LLM
        assert explanation_handle is not None
        assert explanation_grammar is not None

        explanation_messages = build_explanation_messages(
            system_prompt=explanation_system_prompt,
            prompt_case=item.prompt_case,
            sanitized_changed_paths=record["sanitized_changed_paths"],
            sanitized_sparse_patch_preview=record["sanitized_sparse_patch_preview"],
            merged_patch_preview=record["merged_patch_preview"],
        )

        generation_started = perf_counter()
        response: dict[str, Any] | None = None
        raw_output = ""
        generation_error: str | None = None
        try:
            response = invoke_chat_completion(
                handle=explanation_handle,
                messages=explanation_messages,
                grammar=explanation_grammar,
                temperature=args.explainer_temperature,
                top_p=args.explainer_top_p,
                repeat_penalty=args.explainer_repeat_penalty,
                max_tokens=args.explainer_max_tokens,
            )
            choice = response["choices"][0]
            raw_output = normalize_assistant_content(choice["message"].get("content"))
        except Exception as exc:  # pragma: no cover - runtime dependent
            generation_error = str(exc)
        generation_ended = perf_counter()

        record["explanation_latency_ms"] = (generation_ended - generation_started) * 1000.0
        record["explanation_ready_ms"] = (generation_ended - item.submitted_perf) * 1000.0
        if record["patch_applied_ms"] is not None:
            record["ui_gap_ms"] = record["explanation_ready_ms"] - record["patch_applied_ms"]
        record["explanation_raw_output"] = raw_output
        record["explanation_trimmed_output"] = raw_output.strip()
        record["explanation_generation_error"] = generation_error

        if response is not None:
            choice = response["choices"][0]
            record["explanation_finish_reason"] = choice.get("finish_reason")
            usage = response.get("usage", {})
            record["explanation_prompt_tokens"] = usage.get("prompt_tokens")
            record["explanation_completion_tokens"] = usage.get("completion_tokens")
            record["explanation_total_tokens"] = usage.get("total_tokens")
            if record["explanation_completion_tokens"] and record["explanation_latency_ms"] and record["explanation_latency_ms"] > 0:
                record["explanation_tokens_per_second"] = record["explanation_completion_tokens"] / (
                    record["explanation_latency_ms"] / 1000.0
                )

        latest_patch_version = shared_state.latest_patch_version(item.prompt_case.stream_id)
        if item.patch_version < latest_patch_version:
            record["explanation_discarded"] = True
            record["explanation_errors"] = [
                f"Explanation discarded because stream {item.prompt_case.stream_id!r} advanced to patch version {latest_patch_version}."
            ]
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        if generation_error is not None:
            record["explanation_errors"] = [f"Explanation generation error: {generation_error}"]
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        parsed_payload, syntax_errors = parse_json_object(raw_output)
        if syntax_errors:
            record["explanation_errors"] = syntax_errors
            completion_queue.put(item)
            shared_state.record_queue_size("completion", completion_queue.qsize())
            continue

        explanation_errors = validate_explanation_payload(parsed_payload, explanation_schema or {})
        record["explanation_payload"] = parsed_payload
        record["explanation_text"] = parsed_payload.get("explanation")
        record["explanation_available"] = not explanation_errors
        record["explanation_pass"] = not explanation_errors
        record["explanation_errors"] = explanation_errors
        completion_queue.put(item)
        shared_state.record_queue_size("completion", completion_queue.qsize())


def load_model_handle(
    *,
    label: str,
    model_path: Path,
    n_ctx: int,
    n_threads: int | None,
    n_gpu_layers: int,
    seed: int,
    chat_format: str | None,
    reporter: StatusReporter,
) -> ModelHandle:
    llama_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "seed": seed,
        "verbose": False,
    }
    if n_threads is not None:
        llama_kwargs["n_threads"] = n_threads
    if chat_format is not None:
        llama_kwargs["chat_format"] = chat_format

    reporter.emit(f"Loading {label} model: {model_path}")
    started = perf_counter()
    llm = Llama(**llama_kwargs)
    load_ms = (perf_counter() - started) * 1000.0
    reporter.emit(f"Loaded {label} model in {load_ms:.2f} ms")
    return ModelHandle(
        label=label,
        model_path=model_path,
        llm=llm,
        lock=threading.Lock(),
        load_ms=load_ms,
    )


def build_summary(records: list[dict[str, Any]], queue_backlog_max: dict[str, int]) -> dict[str, Any]:
    patch_ready_values = [record["patch_ready_ms"] for record in records if record["patch_ready_ms"] is not None]
    patch_applied_values = [
        record["patch_applied_ms"] for record in records if record["patch_applied_ms"] is not None
    ]
    explanation_ready_values = [
        record["explanation_ready_ms"] for record in records if record["explanation_ready_ms"] is not None
    ]
    ui_gap_values = [record["ui_gap_ms"] for record in records if record["ui_gap_ms"] is not None]
    patch_queue_wait_values = [
        record["patch_queue_wait_ms"] for record in records if record["patch_queue_wait_ms"] is not None
    ]
    apply_queue_wait_values = [
        record["apply_queue_wait_ms"] for record in records if record["apply_queue_wait_ms"] is not None
    ]
    explanation_queue_wait_values = [
        record["explanation_queue_wait_ms"]
        for record in records
        if record["explanation_queue_wait_ms"] is not None
    ]
    patch_latency_values = [record["patch_latency_ms"] for record in records if record["patch_latency_ms"] is not None]
    apply_latency_values = [record["apply_latency_ms"] for record in records if record["apply_latency_ms"] is not None]
    explanation_latency_values = [
        record["explanation_latency_ms"] for record in records if record["explanation_latency_ms"] is not None
    ]
    patch_tps_values = [
        record["patch_tokens_per_second"]
        for record in records
        if record["patch_tokens_per_second"] is not None
    ]
    explanation_tps_values = [
        record["explanation_tokens_per_second"]
        for record in records
        if record["explanation_tokens_per_second"] is not None
    ]
    raw_changed_counts = [record["raw_changed_param_count"] for record in records]
    sanitized_changed_counts = [record["sanitized_changed_param_count"] for record in records]
    raw_null_ratios = [record["raw_null_ratio"] for record in records if record["raw_null_ratio"] is not None]
    sanitized_null_ratios = [
        record["sanitized_null_ratio"] for record in records if record["sanitized_null_ratio"] is not None
    ]
    raw_omitted_ratios = [
        record["raw_omitted_ratio"] for record in records if record["raw_omitted_ratio"] is not None
    ]
    sanitized_omitted_ratios = [
        record["sanitized_omitted_ratio"]
        for record in records
        if record["sanitized_omitted_ratio"] is not None
    ]

    clamp_frequency: dict[str, int] = defaultdict(int)
    for record in records:
        for event in record["clamp_events"]:
            clamp_frequency[event["path"]] += 1

    raw_sparse_exact_values = [
        record["raw_sparse_fidelity"]["sparse_exact_match"]
        for record in records
        if record["raw_sparse_fidelity"]["sparse_exact_match"] is not None
    ]
    sanitized_sparse_exact_values = [
        record["sanitized_sparse_fidelity"]["sparse_exact_match"]
        for record in records
        if record["sanitized_sparse_fidelity"]["sparse_exact_match"] is not None
    ]
    clamp_expectation_values = [
        record["clamp_expectation_pass"]
        for record in records
        if record["clamp_expectation_pass"] is not None
    ]
    raw_noop_exact_values = [
        record["raw_sparse_fidelity"]["noop_exact_pass"]
        for record in records
        if record["raw_sparse_fidelity"]["noop_exact_pass"] is not None
    ]
    sanitized_noop_exact_values = [
        record["sanitized_sparse_fidelity"]["noop_exact_pass"]
        for record in records
        if record["sanitized_sparse_fidelity"]["noop_exact_pass"] is not None
    ]

    summary = {
        "total_runs": len(records),
        "raw_syntax_pass_count": sum(1 for record in records if record["raw_syntax_pass"]),
        "raw_schema_pass_count": sum(1 for record in records if record["raw_schema_pass"]),
        "raw_key_order_pass_count": sum(1 for record in records if record["raw_key_order_pass"]),
        "raw_range_pass_count": sum(1 for record in records if record["raw_range_pass"]),
        "raw_semantic_pass_count": sum(1 for record in records if record["raw_semantic_pass"]),
        "post_clamp_apply_pass_count": sum(1 for record in records if record["post_clamp_apply_pass"]),
        "patch_pipeline_pass_count": sum(1 for record in records if record["patch_pipeline_pass"]),
        "explanation_requested_count": sum(1 for record in records if record["explanation_requested"]),
        "explanation_available_count": sum(1 for record in records if record["explanation_available"]),
        "explanation_pass_count": sum(1 for record in records if record["explanation_pass"]),
        "explanation_cancelled_count": sum(1 for record in records if record["explanation_cancelled"]),
        "explanation_discard_count": sum(1 for record in records if record["explanation_discarded"]),
        "clamp_intervention_count": sum(1 for record in records if record["clamp_intervention"]),
        "clamp_event_count_total": sum(record["clamp_event_count"] for record in records),
        "raw_false_edit_count_total": sum(
            record["raw_sparse_fidelity"]["false_edit_count"] for record in records
        ),
        "sanitized_false_edit_count_total": sum(
            record["sanitized_sparse_fidelity"]["false_edit_count"] for record in records
        ),
        "latency_ms": {
            "patch_ready": summarize_metric(patch_ready_values),
            "patch_applied": summarize_metric(patch_applied_values),
            "explanation_ready": summarize_metric(explanation_ready_values),
            "ui_gap": summarize_metric(ui_gap_values),
            "patch_queue_wait": summarize_metric(patch_queue_wait_values),
            "apply_queue_wait": summarize_metric(apply_queue_wait_values),
            "explanation_queue_wait": summarize_metric(explanation_queue_wait_values),
            "patch_generation": summarize_metric(patch_latency_values),
            "apply": summarize_metric(apply_latency_values),
            "explanation_generation": summarize_metric(explanation_latency_values),
        },
        "throughput": {
            "patch_tokens_per_second": summarize_metric(patch_tps_values),
            "explanation_tokens_per_second": summarize_metric(explanation_tps_values),
        },
        "changed_param_count": {
            "raw_mean": statistics.fmean(raw_changed_counts) if raw_changed_counts else None,
            "raw_max": max(raw_changed_counts) if raw_changed_counts else None,
            "sanitized_mean": statistics.fmean(sanitized_changed_counts) if sanitized_changed_counts else None,
            "sanitized_max": max(sanitized_changed_counts) if sanitized_changed_counts else None,
        },
        "null_ratio": {
            "raw_mean": statistics.fmean(raw_null_ratios) if raw_null_ratios else None,
            "raw_max": max(raw_null_ratios) if raw_null_ratios else None,
            "sanitized_mean": statistics.fmean(sanitized_null_ratios) if sanitized_null_ratios else None,
            "sanitized_max": max(sanitized_null_ratios) if sanitized_null_ratios else None,
        },
        "omitted_ratio": {
            "raw_mean": statistics.fmean(raw_omitted_ratios) if raw_omitted_ratios else None,
            "raw_max": max(raw_omitted_ratios) if raw_omitted_ratios else None,
            "sanitized_mean": statistics.fmean(sanitized_omitted_ratios) if sanitized_omitted_ratios else None,
            "sanitized_max": max(sanitized_omitted_ratios) if sanitized_omitted_ratios else None,
        },
        "queue_backlog_max": queue_backlog_max,
        "clamp_intervention_rate": (
            sum(1 for record in records if record["clamp_intervention"]) / len(records)
            if records
            else 0.0
        ),
        "clamp_expectation_pass_rate": (
            sum(1 for value in clamp_expectation_values if value) / len(clamp_expectation_values)
        )
        if clamp_expectation_values
        else None,
        "mean_clamp_events_per_prompt": (
            statistics.fmean([record["clamp_event_count"] for record in records]) if records else None
        ),
        "per_path_clamp_frequency": dict(sorted(clamp_frequency.items())),
        "sparse_exact_match_rate": {
            "raw": (sum(1 for value in raw_sparse_exact_values if value) / len(raw_sparse_exact_values))
            if raw_sparse_exact_values
            else None,
            "sanitized": (
                sum(1 for value in sanitized_sparse_exact_values if value) / len(sanitized_sparse_exact_values)
            )
            if sanitized_sparse_exact_values
            else None,
        },
        "noop_exact_pass_rate": {
            "raw": (sum(1 for value in raw_noop_exact_values if value) / len(raw_noop_exact_values))
            if raw_noop_exact_values
            else None,
            "sanitized": (
                sum(1 for value in sanitized_noop_exact_values if value) / len(sanitized_noop_exact_values)
            )
            if sanitized_noop_exact_values
            else None,
        },
        "failure_categories": {},
    }

    if records:
        summary["raw_syntax_pass_rate"] = summary["raw_syntax_pass_count"] / len(records)
        summary["raw_schema_pass_rate"] = summary["raw_schema_pass_count"] / len(records)
        summary["raw_key_order_pass_rate"] = summary["raw_key_order_pass_count"] / len(records)
        summary["raw_range_pass_rate"] = summary["raw_range_pass_count"] / len(records)
        summary["raw_semantic_pass_rate"] = summary["raw_semantic_pass_count"] / len(records)
        summary["post_clamp_apply_pass_rate"] = summary["post_clamp_apply_pass_count"] / len(records)
        summary["patch_pipeline_pass_rate"] = summary["patch_pipeline_pass_count"] / len(records)
        explanation_requested = summary["explanation_requested_count"]
        summary["explanation_available_rate"] = (
            summary["explanation_available_count"] / explanation_requested if explanation_requested else None
        )
        summary["explanation_pass_rate"] = (
            summary["explanation_pass_count"] / explanation_requested if explanation_requested else None
        )
    else:
        summary["raw_syntax_pass_rate"] = 0.0
        summary["raw_schema_pass_rate"] = 0.0
        summary["raw_key_order_pass_rate"] = 0.0
        summary["raw_range_pass_rate"] = 0.0
        summary["raw_semantic_pass_rate"] = 0.0
        summary["post_clamp_apply_pass_rate"] = 0.0
        summary["patch_pipeline_pass_rate"] = 0.0
        summary["explanation_available_rate"] = None
        summary["explanation_pass_rate"] = None

    for record in records:
        category = classify_record(record)
        summary["failure_categories"][category] = summary["failure_categories"].get(category, 0) + 1

    return summary


def classify_record(record: dict[str, Any]) -> str:
    if record["patch_generation_error"]:
        return "patch_generation_error"
    if not record["raw_syntax_pass"]:
        return "patch_syntax_fail"
    if not record["raw_schema_pass"]:
        return "patch_schema_fail"
    if not record["raw_key_order_pass"]:
        return "patch_key_order_fail"
    if not record["raw_semantic_pass"]:
        return "patch_semantic_fail"
    if not record["raw_range_pass"] and not record["apply_success"]:
        return "raw_range_fail_apply_fail"
    if not record["raw_range_pass"]:
        return "raw_range_fail_clamped"
    if not record["apply_success"]:
        return "apply_fail"
    if record["explanation_cancelled"]:
        return "explanation_cancelled"
    if record["explanation_discarded"]:
        return "explanation_discarded"
    if record["explanation_generation_error"]:
        return "explanation_generation_error"
    if record["explanation_requested"] and not record["explanation_pass"]:
        return "explanation_fail"
    if record["clamp_intervention"]:
        return "pass_with_clamp"
    return "pass"


def build_batch_payload(
    *,
    run_id: str,
    run_state: str,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    results_path: Path,
    latency_path: Path,
    status_path: Path,
    clamp_events_path: Path,
    patch_handle: ModelHandle,
    explanation_handle: ModelHandle | None,
    planned_runs: int,
) -> dict[str, Any]:
    return {
        "experiment": "Experiment 1.2: Patch-Only Async Pipeline Sandbox",
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
            "clamp_events_jsonl": str(clamp_events_path),
        },
        "config": serialize_args(args),
        "models": {
            "patch_model": {
                "model_path": str(patch_handle.model_path),
                "model_name": patch_handle.model_path.name,
                "model_file_size": format_bytes(patch_handle.model_path.stat().st_size),
                "model_load_ms": patch_handle.load_ms,
            },
            "explainer_model": (
                None
                if explanation_handle is None
                else {
                    "model_path": str(explanation_handle.model_path),
                    "model_name": explanation_handle.model_path.name,
                    "model_file_size": format_bytes(explanation_handle.model_path.stat().st_size),
                    "model_load_ms": explanation_handle.load_ms,
                    "shared_with_patch": explanation_handle.shared_with_patch,
                }
            ),
        },
        "progress": {
            "completed_runs": len(records),
            "total_runs": planned_runs,
        },
        "summary": summary,
        "results": records,
    }


def save_latency_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prompt_id",
        "stream_id",
        "patch_contract",
        "explanation_mode",
        "patch_queue_wait_ms",
        "apply_queue_wait_ms",
        "explanation_queue_wait_ms",
        "patch_latency_ms",
        "patch_parse_latency_ms",
        "patch_validation_latency_ms",
        "apply_latency_ms",
        "explanation_latency_ms",
        "patch_ready_ms",
        "patch_applied_ms",
        "explanation_ready_ms",
        "ui_gap_ms",
        "raw_syntax_pass",
        "raw_schema_pass",
        "raw_key_order_pass",
        "raw_range_pass",
        "raw_semantic_pass",
        "apply_success",
        "patch_pipeline_pass",
        "clamp_event_count",
        "clamp_intervention",
        "raw_changed_param_count",
        "sanitized_changed_param_count",
        "raw_null_ratio",
        "sanitized_null_ratio",
        "raw_omitted_ratio",
        "sanitized_omitted_ratio",
        "patch_prompt_tokens",
        "patch_completion_tokens",
        "patch_total_tokens",
        "patch_tokens_per_second",
        "explanation_prompt_tokens",
        "explanation_completion_tokens",
        "explanation_total_tokens",
        "explanation_tokens_per_second",
        "patch_finish_reason",
        "explanation_finish_reason",
        "patch_generation_error",
        "explanation_generation_error",
        "raw_validation_errors",
        "apply_errors",
        "explanation_errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda value: value["submitted_index"]):
            writer.writerow(
                {
                    "prompt_id": record["prompt_id"],
                    "stream_id": record["stream_id"],
                    "patch_contract": record["patch_contract"],
                    "explanation_mode": record["explanation_mode"],
                    "patch_queue_wait_ms": format_optional_float(record["patch_queue_wait_ms"]),
                    "apply_queue_wait_ms": format_optional_float(record["apply_queue_wait_ms"]),
                    "explanation_queue_wait_ms": format_optional_float(record["explanation_queue_wait_ms"]),
                    "patch_latency_ms": format_optional_float(record["patch_latency_ms"]),
                    "patch_parse_latency_ms": format_optional_float(record["patch_parse_latency_ms"]),
                    "patch_validation_latency_ms": format_optional_float(record["patch_validation_latency_ms"]),
                    "apply_latency_ms": format_optional_float(record["apply_latency_ms"]),
                    "explanation_latency_ms": format_optional_float(record["explanation_latency_ms"]),
                    "patch_ready_ms": format_optional_float(record["patch_ready_ms"]),
                    "patch_applied_ms": format_optional_float(record["patch_applied_ms"]),
                    "explanation_ready_ms": format_optional_float(record["explanation_ready_ms"]),
                    "ui_gap_ms": format_optional_float(record["ui_gap_ms"]),
                    "raw_syntax_pass": record["raw_syntax_pass"],
                    "raw_schema_pass": record["raw_schema_pass"],
                    "raw_key_order_pass": record["raw_key_order_pass"],
                    "raw_range_pass": record["raw_range_pass"],
                    "raw_semantic_pass": record["raw_semantic_pass"],
                    "apply_success": record["apply_success"],
                    "patch_pipeline_pass": record["patch_pipeline_pass"],
                    "clamp_event_count": record["clamp_event_count"],
                    "clamp_intervention": record["clamp_intervention"],
                    "raw_changed_param_count": record["raw_changed_param_count"],
                    "sanitized_changed_param_count": record["sanitized_changed_param_count"],
                    "raw_null_ratio": format_optional_float(record["raw_null_ratio"], precision=6),
                    "sanitized_null_ratio": format_optional_float(record["sanitized_null_ratio"], precision=6),
                    "raw_omitted_ratio": format_optional_float(record["raw_omitted_ratio"], precision=6),
                    "sanitized_omitted_ratio": format_optional_float(record["sanitized_omitted_ratio"], precision=6),
                    "patch_prompt_tokens": record["patch_prompt_tokens"] or "",
                    "patch_completion_tokens": record["patch_completion_tokens"] or "",
                    "patch_total_tokens": record["patch_total_tokens"] or "",
                    "patch_tokens_per_second": format_optional_float(record["patch_tokens_per_second"]),
                    "explanation_prompt_tokens": record["explanation_prompt_tokens"] or "",
                    "explanation_completion_tokens": record["explanation_completion_tokens"] or "",
                    "explanation_total_tokens": record["explanation_total_tokens"] or "",
                    "explanation_tokens_per_second": format_optional_float(record["explanation_tokens_per_second"]),
                    "patch_finish_reason": record["patch_finish_reason"] or "",
                    "explanation_finish_reason": record["explanation_finish_reason"] or "",
                    "patch_generation_error": record["patch_generation_error"] or "",
                    "explanation_generation_error": record["explanation_generation_error"] or "",
                    "raw_validation_errors": " | ".join(record["raw_validation_errors"]),
                    "apply_errors": " | ".join(record["apply_errors"]),
                    "explanation_errors": " | ".join(record["explanation_errors"]),
                }
            )


def save_clamp_events_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda value: value["submitted_index"]):
            for event in record["clamp_events"]:
                payload = {
                    "prompt_id": record["prompt_id"],
                    "stream_id": record["stream_id"],
                    **event,
                }
                handle.write(json.dumps(payload))
                handle.write("\n")


def format_optional_float(value: float | None, precision: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def write_batch_artifacts(
    *,
    run_id: str,
    run_state: str,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    results_path: Path,
    latency_path: Path,
    status_path: Path,
    clamp_events_path: Path,
    patch_handle: ModelHandle,
    explanation_handle: ModelHandle | None,
    queue_backlog_max: dict[str, int],
    planned_runs: int,
) -> dict[str, Any]:
    ordered_records = sorted(records, key=lambda value: value["submitted_index"])
    summary = build_summary(ordered_records, queue_backlog_max)
    payload = build_batch_payload(
        run_id=run_id,
        run_state=run_state,
        args=args,
        records=ordered_records,
        summary=summary,
        results_path=results_path,
        latency_path=latency_path,
        status_path=status_path,
        clamp_events_path=clamp_events_path,
        patch_handle=patch_handle,
        explanation_handle=explanation_handle,
        planned_runs=planned_runs,
    )
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_latency_csv(latency_path, ordered_records)
    save_clamp_events_jsonl(clamp_events_path, ordered_records)
    return summary


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    patch_grammar_path, patch_schema_path = resolve_patch_contract_paths(args)

    ensure_file(args.patch_model_path, "Patch model file")
    ensure_file(args.parameter_registry_file, "Parameter registry file")
    ensure_file(patch_grammar_path, "Patch grammar file")
    ensure_file(patch_schema_path, "Patch schema file")
    ensure_file(args.prompts_file, "Prompt fixture file")
    if args.few_shot_file is not None:
        ensure_file(args.few_shot_file, "Few-shot file")
    if args.explanation_mode == EXPLANATION_MODE_LLM:
        ensure_file(args.explanation_grammar_file, "Explanation grammar file")
        ensure_file(args.explanation_schema_file, "Explanation schema file")

    if args.explanation_runtime_mode == EXPLANATION_RUNTIME_SERIALIZED and args.explanation_mode != EXPLANATION_MODE_LLM:
        raise ValueError("Serialized explanation runtime mode is only valid when --explanation-mode llm.")
    if (
        args.explanation_runtime_mode == EXPLANATION_RUNTIME_SERIALIZED
        and args.explainer_model_path is not None
        and args.explainer_model_path != args.patch_model_path
    ):
        raise ValueError(
            "--explanation-runtime-mode serialized can only be used when --explainer-model-path is omitted or matches --patch-model-path."
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = args.results_dir / f"run_{run_id}.json"
    latency_path = args.logs_dir / f"latency_{run_id}.csv"
    clamp_events_path = args.logs_dir / f"clamp_events_{run_id}.jsonl"
    status_path = args.status_file or (args.logs_dir / f"status_{run_id}.log")
    reporter = StatusReporter(status_path)

    reporter.emit("Starting Experiment 1.2: Patch-Only Async Pipeline Sandbox")
    reporter.emit(f"Run ID: {run_id}")
    reporter.emit(f"Patch contract: {args.patch_contract}")
    reporter.emit(f"Explanation mode: {args.explanation_mode}")
    reporter.emit(f"Patch model path: {args.patch_model_path}")
    reporter.emit(f"Results JSON: {results_path}")
    reporter.emit(f"Latency CSV: {latency_path}")
    reporter.emit(f"Clamp events JSONL: {clamp_events_path}")
    reporter.emit(f"Status log: {status_path}")

    registry = load_parameter_registry(args.parameter_registry_file)
    patch_schema = load_json(patch_schema_path)
    explanation_schema = load_json(args.explanation_schema_file) if args.explanation_mode == EXPLANATION_MODE_LLM else None
    prompt_cases = load_prompt_cases(args.prompts_file, selected_ids=set(args.prompt_id), limit=args.limit)
    few_shot_examples = load_few_shot_examples(args.few_shot_file, args.few_shot_count)

    patch_system_prompt = load_patch_system_prompt(
        args.patch_system_prompt_file,
        args.patch_contract,
        registry,
    )
    explanation_system_prompt = load_explanation_system_prompt(args.explanation_system_prompt_file)

    reporter.emit(f"Selected prompt cases: {len(prompt_cases)}")
    reporter.emit(f"Few-shot examples loaded: {len(few_shot_examples)}")

    reporter.emit(f"Validating patch grammar file: {patch_grammar_path}")
    validate_gbnf_text(load_text(patch_grammar_path))
    patch_grammar = LlamaGrammar.from_file(str(patch_grammar_path))

    explanation_grammar = None
    if args.explanation_mode == EXPLANATION_MODE_LLM:
        reporter.emit(f"Validating explanation grammar file: {args.explanation_grammar_file}")
        validate_gbnf_text(load_text(args.explanation_grammar_file))
        explanation_grammar = LlamaGrammar.from_file(str(args.explanation_grammar_file))
        if not isinstance(explanation_schema, dict):
            raise ValueError("Explanation schema must be a JSON object.")

    if args.patch_n_gpu_layers <= 0 and args.explanation_n_gpu_layers <= 0:
        reporter.emit("Inference mode: CPU-only.")
    else:
        gpu_runtime_visible = detect_nvidia_runtime_visible()
        llama_gpu_support = detect_llama_gpu_support()
        if not gpu_runtime_visible:
            reporter.emit("Inference mode warning: GPU offload requested but no Nvidia runtime is visible.")
        elif llama_gpu_support is False:
            reporter.emit(
                "Inference mode warning: GPU offload requested but the installed llama.cpp backend does not report GPU offload support."
            )
        else:
            reporter.emit("Inference mode: GPU offload requested.")

    patch_handle = load_model_handle(
        label="patch",
        model_path=args.patch_model_path,
        n_ctx=args.patch_n_ctx,
        n_threads=args.patch_n_threads,
        n_gpu_layers=args.patch_n_gpu_layers,
        seed=args.seed,
        chat_format=args.patch_chat_format,
        reporter=reporter,
    )

    explanation_handle: ModelHandle | None = None
    if args.explanation_mode == EXPLANATION_MODE_LLM:
        if args.explanation_runtime_mode == EXPLANATION_RUNTIME_SERIALIZED:
            explanation_handle = ModelHandle(
                label="explainer",
                model_path=patch_handle.model_path,
                llm=patch_handle.llm,
                lock=patch_handle.lock,
                load_ms=0.0,
                shared_with_patch=True,
            )
            reporter.emit("Explainer runtime mode: serialized through the patch model instance.")
        else:
            explainer_model_path = args.explainer_model_path or args.patch_model_path
            ensure_file(explainer_model_path, "Explainer model file")
            explanation_handle = load_model_handle(
                label="explainer",
                model_path=explainer_model_path,
                n_ctx=args.explainer_n_ctx,
                n_threads=args.explainer_n_threads,
                n_gpu_layers=args.explainer_n_gpu_layers,
                seed=args.seed + 1,
                chat_format=args.explainer_chat_format,
                reporter=reporter,
            )
            reporter.emit("Explainer runtime mode: isolated model instance.")

    records: list[dict[str, Any]] = []
    shared_state = SharedPipelineState()
    write_batch_artifacts(
        run_id=run_id,
        run_state="running",
        args=args,
        records=records,
        results_path=results_path,
        latency_path=latency_path,
        status_path=status_path,
        clamp_events_path=clamp_events_path,
        patch_handle=patch_handle,
        explanation_handle=explanation_handle,
        queue_backlog_max=shared_state.max_queue_sizes(),
        planned_runs=len(prompt_cases),
    )

    patch_input_queue: queue.Queue[PipelineJob | object] = queue.Queue(maxsize=args.worker_queue_size)
    apply_queue: queue.Queue[PipelineJob | object] = queue.Queue(maxsize=args.worker_queue_size)
    explanation_queue: queue.Queue[PipelineJob | object] | None = None
    if args.explanation_mode != EXPLANATION_MODE_OFF:
        explanation_queue = queue.Queue(maxsize=args.worker_queue_size)
    completion_queue: queue.Queue[PipelineJob | object] = queue.Queue()

    patch_thread = threading.Thread(
        target=patch_worker_loop,
        kwargs={
            "reporter": reporter,
            "input_queue": patch_input_queue,
            "apply_queue": apply_queue,
            "completion_queue": completion_queue,
            "shared_state": shared_state,
            "patch_handle": patch_handle,
            "patch_grammar": patch_grammar,
            "patch_schema": patch_schema,
            "patch_system_prompt": patch_system_prompt,
            "patch_contract": args.patch_contract,
            "registry": registry,
            "few_shot_examples": few_shot_examples,
            "args": args,
        },
        name="patch-worker",
        daemon=True,
    )
    apply_thread = threading.Thread(
        target=apply_worker_loop,
        kwargs={
            "reporter": reporter,
            "apply_queue": apply_queue,
            "explanation_queue": explanation_queue,
            "completion_queue": completion_queue,
            "shared_state": shared_state,
            "registry": registry,
            "explanation_mode": args.explanation_mode,
        },
        name="apply-worker",
        daemon=True,
    )
    explanation_thread = None
    if explanation_queue is not None:
        explanation_thread = threading.Thread(
            target=explanation_worker_loop,
            kwargs={
                "reporter": reporter,
                "explanation_queue": explanation_queue,
                "completion_queue": completion_queue,
                "shared_state": shared_state,
                "explanation_mode": args.explanation_mode,
                "explanation_handle": explanation_handle,
                "explanation_grammar": explanation_grammar,
                "explanation_schema": explanation_schema,
                "explanation_system_prompt": explanation_system_prompt,
                "args": args,
            },
            name="explanation-worker",
            daemon=True,
        )

    patch_thread.start()
    apply_thread.start()
    if explanation_thread is not None:
        explanation_thread.start()

    reporter.emit("Submitting prompt cases into the async pipeline.")
    for submitted_index, prompt_case in enumerate(prompt_cases, start=1):
        record = initialize_record(
            prompt_case=prompt_case,
            patch_contract=args.patch_contract,
            explanation_mode=args.explanation_mode,
            submitted_index=submitted_index,
        )
        job = PipelineJob(
            prompt_case=prompt_case,
            record=record,
            submitted_perf=perf_counter(),
            patch_queue_entered_perf=perf_counter(),
        )
        patch_input_queue.put(job)
        shared_state.record_queue_size("patch_input", patch_input_queue.qsize())

    patch_input_queue.put(SENTINEL)
    shared_state.record_queue_size("patch_input", patch_input_queue.qsize())

    try:
        while True:
            completed = completion_queue.get()
            if completed is SENTINEL:
                break
            assert isinstance(completed, PipelineJob)
            records.append(completed.record)
            summary = write_batch_artifacts(
                run_id=run_id,
                run_state="running",
                args=args,
                records=records,
                results_path=results_path,
                latency_path=latency_path,
                status_path=status_path,
                clamp_events_path=clamp_events_path,
                patch_handle=patch_handle,
                explanation_handle=explanation_handle,
                queue_backlog_max=shared_state.max_queue_sizes(),
                planned_runs=len(prompt_cases),
            )
            reporter.emit(
                "Completed "
                f"[{completed.record['prompt_id']}] | "
                f"patch_pass={completed.record['patch_pipeline_pass']} | "
                f"raw_range_pass={completed.record['raw_range_pass']} | "
                f"clamps={completed.record['clamp_event_count']} | "
                f"patch_ready={format_optional_float(completed.record['patch_ready_ms'])} ms | "
                f"patch_applied={format_optional_float(completed.record['patch_applied_ms'])} ms | "
                f"explanation_ready={format_optional_float(completed.record['explanation_ready_ms'])} ms"
            )
            reporter.emit(
                f"Progress: {len(records)}/{len(prompt_cases)} completed so far; patch pipeline pass rate {summary['patch_pipeline_pass_rate']:.1%}"
            )
    except KeyboardInterrupt:
        summary = write_batch_artifacts(
            run_id=run_id,
            run_state="interrupted",
            args=args,
            records=records,
            results_path=results_path,
            latency_path=latency_path,
            status_path=status_path,
            clamp_events_path=clamp_events_path,
            patch_handle=patch_handle,
            explanation_handle=explanation_handle,
            queue_backlog_max=shared_state.max_queue_sizes(),
            planned_runs=len(prompt_cases),
        )
        reporter.emit("Run interrupted by user. Partial artifacts were written before exit.")
        reporter.emit(f"Completed prompts before interruption: {summary['total_runs']}")
        return 130

    summary = write_batch_artifacts(
        run_id=run_id,
        run_state="completed",
        args=args,
        records=records,
        results_path=results_path,
        latency_path=latency_path,
        status_path=status_path,
        clamp_events_path=clamp_events_path,
        patch_handle=patch_handle,
        explanation_handle=explanation_handle,
        queue_backlog_max=shared_state.max_queue_sizes(),
        planned_runs=len(prompt_cases),
    )

    reporter.emit("Async pipeline completed.")
    reporter.emit(
        f"Patch pipeline pass rate: {summary['patch_pipeline_pass_count']}/{summary['total_runs']} ({summary['patch_pipeline_pass_rate']:.1%})"
    )
    reporter.emit(
        f"Raw range pass rate: {summary['raw_range_pass_count']}/{summary['total_runs']} ({summary['raw_range_pass_rate']:.1%})"
    )
    reporter.emit(
        f"Clamp intervention rate: {summary['clamp_intervention_count']}/{summary['total_runs']} ({summary['clamp_intervention_rate']:.1%})"
    )
    if summary["latency_ms"]["patch_ready"]["mean"] is not None:
        reporter.emit(f"Mean patch ready latency: {summary['latency_ms']['patch_ready']['mean']:.2f} ms")
    if summary["latency_ms"]["patch_applied"]["mean"] is not None:
        reporter.emit(f"Mean patch applied latency: {summary['latency_ms']['patch_applied']['mean']:.2f} ms")
    if summary["latency_ms"]["explanation_ready"]["mean"] is not None:
        reporter.emit(
            f"Mean explanation ready latency: {summary['latency_ms']['explanation_ready']['mean']:.2f} ms"
        )
    reporter.emit(f"Results JSON written to: {results_path}")
    reporter.emit(f"Latency CSV written to: {latency_path}")
    reporter.emit(f"Clamp events JSONL written to: {clamp_events_path}")
    reporter.emit(f"Status log written to: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
