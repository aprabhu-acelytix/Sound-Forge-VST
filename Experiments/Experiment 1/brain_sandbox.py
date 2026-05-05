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
        "\"Experiments/Experiment 1/requirements.txt\"'."
    ) from exc


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = EXPERIMENT_ROOT / "prompts" / "system_prompt_v1.txt"
DEFAULT_GRAMMAR_FILE = EXPERIMENT_ROOT / "grammars" / "synth_params_v1.gbnf"
DEFAULT_SCHEMA_FILE = EXPERIMENT_ROOT / "schemas" / "synth_params_v1.json"
DEFAULT_PROMPTS_FILE = EXPERIMENT_ROOT / "fixtures" / "test_prompts.json"
DEFAULT_RESULTS_DIR = EXPERIMENT_ROOT / "results"
DEFAULT_LOGS_DIR = EXPERIMENT_ROOT / "logs"


@dataclass
class PromptCase:
    case_id: str
    prompt: str
    tags: list[str]


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
        description="Run the Sound Forge Brain Sandbox against a local GGUF model."
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
        default=96,
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

        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Every prompt case needs a non-empty string 'id': {item!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Every prompt case needs a non-empty string 'prompt': {item!r}")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"Prompt case 'tags' must be a list of strings: {item!r}")

        if selected_ids and case_id not in selected_ids:
            continue

        loaded_cases.append(PromptCase(case_id=case_id, prompt=prompt.strip(), tags=tags))

    if limit is not None:
        loaded_cases = loaded_cases[:limit]

    if not loaded_cases:
        raise ValueError("No prompt cases matched the selected filters.")

    return loaded_cases


def load_schema(path: Path) -> dict[str, Any]:
    schema = load_json(path)
    if not isinstance(schema, dict):
        raise ValueError("Schema root must be an object.")

    properties = schema.get("properties")
    required = schema.get("required")
    if schema.get("type") != "object":
        raise ValueError("Schema root type must be 'object'.")
    if not isinstance(properties, dict) or not properties:
        raise ValueError("Schema must define non-empty 'properties'.")
    if not isinstance(required, list) or not required:
        raise ValueError("Schema must define a non-empty 'required' list.")

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


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema["properties"]
    expected_keys = list(properties.keys())
    required_keys = schema["required"]
    actual_keys = list(payload.keys())

    schema_errors: list[str] = []
    range_errors: list[str] = []

    missing_keys = [key for key in required_keys if key not in payload]
    extra_keys = [key for key in actual_keys if key not in properties]
    key_order_pass = actual_keys == expected_keys

    if missing_keys:
        schema_errors.append(f"Missing required keys: {', '.join(missing_keys)}.")
    if extra_keys:
        schema_errors.append(f"Unexpected keys: {', '.join(extra_keys)}.")
    if not key_order_pass:
        schema_errors.append(
            "Key order mismatch: expected "
            f"{expected_keys}, received {actual_keys}."
        )

    for key in expected_keys:
        if key not in payload:
            continue

        spec = properties[key]
        value = payload[key]
        expected_type = spec.get("type")

        if expected_type == "string":
            if not isinstance(value, str):
                schema_errors.append(f"{key} must be a string, received {type(value).__name__}.")
                continue

            enum_values = spec.get("enum", [])
            if enum_values and value not in enum_values:
                schema_errors.append(
                    f"{key} must be one of {enum_values}, received {value!r}."
                )
            continue

        if expected_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                schema_errors.append(f"{key} must be a number, received {type(value).__name__}.")
                continue

            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                schema_errors.append(f"{key} must be finite, received {value!r}.")
                continue

            minimum = spec.get("minimum")
            maximum = spec.get("maximum")

            if minimum is not None and numeric_value < float(minimum):
                range_errors.append(
                    f"{key} is below minimum {minimum}: received {numeric_value}."
                )
            if maximum is not None and numeric_value > float(maximum):
                range_errors.append(
                    f"{key} is above maximum {maximum}: received {numeric_value}."
                )
            continue

        schema_errors.append(f"{key} uses unsupported schema type {expected_type!r}.")

    return {
        "schema_pass": not schema_errors,
        "range_pass": not range_errors,
        "key_order_pass": key_order_pass,
        "schema_errors": schema_errors,
        "range_errors": range_errors,
    }


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
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


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
        "experiment": "Experiment 1: The Brain Sandbox",
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
    return "pass"


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["latency_ms"] for record in records if record["latency_ms"] is not None]
    syntax_passes = sum(1 for record in records if record["syntax_pass"])
    schema_passes = sum(1 for record in records if record["schema_pass"])
    range_passes = sum(1 for record in records if record["range_pass"])
    key_order_passes = sum(1 for record in records if record["key_order_pass"])
    overall_passes = sum(1 for record in records if record["overall_pass"])
    failure_categories: dict[str, int] = {}

    for record in records:
        category = classify_record(record)
        failure_categories[category] = failure_categories.get(category, 0) + 1

    summary = {
        "total_runs": len(records),
        "syntax_pass_count": syntax_passes,
        "schema_pass_count": schema_passes,
        "range_pass_count": range_passes,
        "key_order_pass_count": key_order_passes,
        "overall_pass_count": overall_passes,
        "syntax_pass_rate": (syntax_passes / len(records)) if records else 0.0,
        "schema_pass_rate": (schema_passes / len(records)) if records else 0.0,
        "range_pass_rate": (range_passes / len(records)) if records else 0.0,
        "key_order_pass_rate": (key_order_passes / len(records)) if records else 0.0,
        "overall_pass_rate": (overall_passes / len(records)) if records else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "failure_categories": failure_categories,
    }
    return summary


def save_latency_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "prompt_id",
        "latency_ms",
        "syntax_pass",
        "schema_pass",
        "range_pass",
        "key_order_pass",
        "overall_pass",
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
                    "syntax_pass": record["syntax_pass"],
                    "schema_pass": record["schema_pass"],
                    "range_pass": record["range_pass"],
                    "key_order_pass": record["key_order_pass"],
                    "overall_pass": record["overall_pass"],
                    "prompt_tokens": record["prompt_tokens"] or "",
                    "completion_tokens": record["completion_tokens"] or "",
                    "total_tokens": record["total_tokens"] or "",
                    "tokens_per_second": (
                        f"{record['tokens_per_second']:.3f}"
                        if record["tokens_per_second"] is not None
                        else ""
                    ),
                    "finish_reason": record["finish_reason"] or "",
                    "generation_error": record["generation_error"] or "",
                    "validation_errors": " | ".join(record["validation_errors"]),
                }
            )


def run_prompt_case(
    llm: Llama,
    grammar: LlamaGrammar,
    schema: dict[str, Any],
    system_prompt: str,
    prompt_case: PromptCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_case.prompt},
    ]

    started = perf_counter()
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

    ended = perf_counter()
    latency_ms = (ended - started) * 1000.0

    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    tokens_per_second = None

    if response is not None:
        choice = response["choices"][0]
        finish_reason = choice.get("finish_reason")
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if completion_tokens and latency_ms > 0:
            tokens_per_second = completion_tokens / (latency_ms / 1000.0)

    record: dict[str, Any] = {
        "prompt_id": prompt_case.case_id,
        "prompt": prompt_case.prompt,
        "tags": prompt_case.tags,
        "latency_ms": latency_ms,
        "raw_output": raw_output,
        "trimmed_output": raw_output.strip(),
        "parsed_payload": None,
        "syntax_pass": False,
        "schema_pass": False,
        "range_pass": False,
        "key_order_pass": False,
        "overall_pass": False,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
        "generation_error": generation_error,
        "validation_errors": [],
    }

    if generation_error:
        record["validation_errors"] = [f"Generation error: {generation_error}"]
        return record

    parsed_payload, syntax_errors = parse_json_object(raw_output)
    if syntax_errors:
        record["validation_errors"] = syntax_errors
        return record

    record["syntax_pass"] = True
    record["parsed_payload"] = parsed_payload

    validation = validate_payload(parsed_payload, schema)
    record["schema_pass"] = validation["schema_pass"]
    record["range_pass"] = validation["range_pass"]
    record["key_order_pass"] = validation["key_order_pass"]
    record["validation_errors"] = [
        *validation["schema_errors"],
        *validation["range_errors"],
    ]
    record["overall_pass"] = (
        record["syntax_pass"]
        and record["schema_pass"]
        and record["range_pass"]
        and record["key_order_pass"]
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

    reporter.emit("Starting Experiment 1: The Brain Sandbox")
    reporter.emit(f"Run ID: {run_id}")
    reporter.emit(f"Model path: {args.model_path}")
    reporter.emit(f"Model file size: {format_bytes(args.model_path.stat().st_size)}")
    reporter.emit(f"Results JSON: {results_path}")
    reporter.emit(f"Latency CSV: {latency_path}")
    reporter.emit(f"Status log: {status_path}")

    system_prompt = load_text(args.system_prompt_file)
    schema = load_schema(args.schema_file)
    prompt_cases = load_prompt_cases(
        args.prompts_file,
        selected_ids=set(args.prompt_id),
        limit=args.limit,
    )
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
                f"Inference mode warning: GPU offload requested with {args.n_gpu_layers} layers, "
                "but no Nvidia runtime is visible in this environment."
            )
        elif llama_gpu_support is False:
            reporter.emit(
                f"Inference mode warning: GPU offload requested with {args.n_gpu_layers} layers, "
                "but the installed llama.cpp backend does not report GPU offload support."
            )
        else:
            reporter.emit(
                f"Inference mode: GPU offload requested with {args.n_gpu_layers} layers."
            )

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
    summary = write_batch_artifacts(
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
            reporter.emit(
                f"Stage 2/3: starting prompt {index}/{len(prompt_cases)} [{prompt_case.case_id}]"
            )
            record = run_prompt_case(
                llm=llm,
                grammar=grammar,
                schema=schema,
                system_prompt=system_prompt,
                prompt_case=prompt_case,
                args=args,
            )
            records.append(record)

            summary = write_batch_artifacts(
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
                f"Stage 2/3: completed prompt {index}/{len(prompt_cases)} [{prompt_case.case_id}] | "
                f"{status} | {record['latency_ms']:.2f} ms | "
                f"finish={record['finish_reason'] or 'n/a'} | errors={error_summary}"
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
        f"Overall pass rate: {summary['overall_pass_count']}/{summary['total_runs']} "
        f"({summary['overall_pass_rate']:.1%})"
    )
    reporter.emit(
        f"Syntax pass rate: {summary['syntax_pass_count']}/{summary['total_runs']} "
        f"({summary['syntax_pass_rate']:.1%})"
    )
    reporter.emit(
        f"Range pass rate: {summary['range_pass_count']}/{summary['total_runs']} "
        f"({summary['range_pass_rate']:.1%})"
    )
    if summary["latency_ms"]["mean"] is not None:
        reporter.emit(f"Mean latency: {summary['latency_ms']['mean']:.2f} ms")
    else:
        reporter.emit("Mean latency: n/a")
    if summary["latency_ms"]["p95"] is not None:
        reporter.emit(f"P95 latency: {summary['latency_ms']['p95']:.2f} ms")
    else:
        reporter.emit("P95 latency: n/a")
    reporter.emit(f"Results JSON written to: {results_path}")
    reporter.emit(f"Latency CSV written to: {latency_path}")
    reporter.emit(f"Status log written to: {status_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
