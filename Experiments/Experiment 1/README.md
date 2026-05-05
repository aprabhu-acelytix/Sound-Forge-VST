# Experiment 1: The Brain Sandbox

This experiment proves the control-plane question first: can a local LLM emit one strict JSON object for synth parameters without extra conversational text?

## Goal

- Force the model to emit a single JSON object with exactly five keys.
- Measure inference latency with `time.perf_counter()`.
- Separate syntax success from semantic range success.
- Save machine-readable artifacts for later wiki ingestion and C++ planning.

## Stack

- Python 3.10+
- `llama-cpp-python`
- GGUF instruct model, ideally a small 7B to 8B class model

Recommended starting model:

- `Qwen2.5 7B Instruct` in GGUF format, quantized around `Q4_K_M` or `Q5_K_M`

## Files

- `brain_sandbox.py` - batch runner and validator
- `prompts/system_prompt_v1.txt` - bounded sound-designer system prompt
- `grammars/synth_params_v1.gbnf` - fixed-order grammar for the output object
- `schemas/synth_params_v1.json` - canonical parameter contract
- `fixtures/test_prompts.json` - baseline evaluation set
- `results/` - timestamped batch JSON artifacts
- `logs/` - timestamped latency CSV artifacts

## Install

```bash
pip install -r "Experiments/Experiment 1/requirements.txt"
```

Docker is not strictly required for this experiment. The runner is just a Python script. However, on this project machine the most reliable GPU path is a standard host Docker container with `--gpus all`, not the `docker sandbox` workflow.

## Run

```bash
python "Experiments/Experiment 1/brain_sandbox.py" \
  --model-path "/absolute/path/to/your-model.gguf"
```

The runner now prints live stage updates to stdout and also writes a status log file during execution. Partial JSON and CSV artifacts are written after every completed prompt, so an interrupted run still leaves usable output behind.

## Host GPU Docker Workflow

This is the recommended path for running Experiment 1 with the RTX 4060 outside the sandbox.

### 1. Open the real repo root

From your outer project folder, change into the inner repo folder first:

```powershell
cd "C:\Users\anike\OneDrive\Documents\career\Projects\Sound Forge VST\Sound Forge VST"
```

The helper scripts and artifact paths below assume you are in that inner folder.

### 2. Put your GGUF model somewhere on the host

Recommended:

- keep the GGUF outside OneDrive if it is large
- example path: `C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf`

The helper script mounts the model directory read-only into the container. The model does not need to live inside this repo.

### Recommended models for Experiment 1

Use these in this order:

1. `bartowski/Qwen2.5-3B-Instruct-GGUF`
   File: `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
   Size: about `1.93 GB`
   Use for: the first smoke test, grammar wiring, and fast reruns

2. `bartowski/Qwen2.5-7B-Instruct-GGUF`
   File: `Qwen2.5-7B-Instruct-Q4_K_M.gguf`
   Size: about `4.68 GB`
   Use for: the primary benchmark for this experiment

3. `bartowski/Mistral-7B-Instruct-v0.3-GGUF`
   File: `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`
   Size: about `4.37 GB`
   Use for: a comparison run to see whether structured-output reliability is model-specific

Optional higher-quality follow-up:

- `bartowski/Qwen2.5-7B-Instruct-GGUF`
  File: `Qwen2.5-7B-Instruct-Q5_K_M.gguf`
  Size: about `5.44 GB`
  Use only after the Q4 model is working, because it is larger and slower.

Why these are the right first picks:

- `Qwen2.5` is strong at instruction following and structured outputs.
- `3B` is cheap and fast for plumbing validation.
- `7B` is a better real benchmark for semantic mapping quality.
- `Mistral 7B` gives you a useful second architecture to compare against.

### Download commands for Windows PowerShell

Create a model directory once:

```powershell
New-Item -ItemType Directory -Force "C:\LLMs\models" | Out-Null
```

Download the smoke-test model:

```powershell
curl.exe -L "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf?download=true" -o "C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
```

Download the primary benchmark model:

```powershell
curl.exe -L "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf?download=true" -o "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
```

Download the comparison model:

```powershell
curl.exe -L "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true" -o "C:\LLMs\models\Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
```

After that, your `-ModelPath` values will be:

- `C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- `C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- `C:\LLMs\models\Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`

### 3. Build and run the GPU container

Short smoke test:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& ".\Experiments\Experiment 1\run_experiment_gpu.ps1" `
  -ModelPath "C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf" `
  -GpuLayers 35 `
  -Limit 3 `
  -RebuildImage
```

Run `Set-ExecutionPolicy` as its own command first. If you prefer a one-liner, separate it with a semicolon:

```powershell
Set-ExecutionPolicy -Scope Process Bypass; & ".\Experiments\Experiment 1\run_experiment_gpu.ps1" -ModelPath "C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf" -GpuLayers 35 -Limit 3 -RebuildImage
```

Full fixture run:

```powershell
& ".\Experiments\Experiment 1\run_experiment_gpu.ps1" `
  -ModelPath "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" `
  -GpuLayers 35
```

Full baseline run on the 3B model:

```powershell
& ".\Experiments\Experiment 1\run_experiment_gpu.ps1" `
  -ModelPath "C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf" `
  -GpuLayers 35
```

Selected prompts only:

```powershell
& ".\Experiments\Experiment 1\run_experiment_gpu.ps1" `
  -ModelPath "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" `
  -GpuLayers 35 `
  -PromptId acid_bass_001,glass_pluck_001
```

Notes:

- The image build compiles `llama-cpp-python` with `GGML_CUDA` enabled.
- Use `-RebuildImage` when the Dockerfile or Python dependencies change.
- Omit `-RebuildImage` for normal reruns after the image already exists.
- The first build is large because it pulls a CUDA development image and compiles `llama-cpp-python`.
- If the build fails, the PowerShell helper now stops immediately instead of attempting a broken `docker run` afterward.

### 3B + 7B benchmark matrix

Run both baseline models back-to-back:

```powershell
& ".\Experiments\Experiment 1\run_benchmark_matrix.ps1"
```

Run both baseline models plus the adversarial fixture set:

```powershell
& ".\Experiments\Experiment 1\run_benchmark_matrix.ps1" -IncludeAdversarial
```

The adversarial fixture file is:

- `Experiments\Experiment 1\fixtures\adversarial_prompts.json`

### Summarize completed runs

After you have one or more `run_*.json` files, summarize them with:

```powershell
python ".\Experiments\Experiment 1\summarize_results.py" `
  ".\Experiments\Experiment 1\results\run_20260504T014841Z.json"
```

Or compare several runs at once:

```powershell
python ".\Experiments\Experiment 1\summarize_results.py" `
  ".\Experiments\Experiment 1\results\run_a.json" `
  ".\Experiments\Experiment 1\results\run_b.json"
```

### 4. Watch progress live

You can monitor the pinned status file in a second PowerShell window:

```powershell
Get-Content ".\Experiments\Experiment 1\logs\status_latest.log" -Wait
```

### 5. Artifacts that will remain in the repo

The helper script bind-mounts this repo into the container, so all logs and results are written back to the host filesystem in-place:

- `Experiments\Experiment 1\logs\status_latest.log`
- `Experiments\Experiment 1\logs\latency_<timestamp>.csv`
- `Experiments\Experiment 1\results\run_<timestamp>.json`

When you come back, I will be able to read those files from the same workspace.

### 6. Optional advanced flags

Examples:

```powershell
& ".\Experiments\Experiment 1\run_experiment_gpu.ps1" `
  -ModelPath "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" `
  -GpuLayers 40 `
  -NCtx 4096 `
  -MaxTokens 96 `
  -Temperature 0.0 `
  -TopP 1.0 `
  -RepeatPenalty 1.0
```

If your model needs an explicit chat formatter:

```powershell
& ".\Experiments\Experiment 1\run_experiment_gpu.ps1" `
  -ModelPath "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" `
  -GpuLayers 35 `
  -ChatFormat chatml
```

Useful flags:

- `--limit 3` to run a short smoke test
- `--prompt-id acid_bass_001 --prompt-id glass_pluck_001` to run selected cases
- `--n-gpu-layers 35` to request GPU offload if your local `llama.cpp` build supports it
- `--chat-format <format>` if the GGUF lacks correct chat metadata
- `--status-file "Experiments/Experiment 1/logs/status_latest.log"` to pin the live status output to a known file

## Progress Visibility

At startup the runner reports:

- run ID
- model path and model size
- result, latency, and status artifact paths
- how many prompts were selected
- whether the run is CPU-only or attempting GPU offload

During execution it reports:

- model-load start and completion
- prompt start and completion for each case
- per-prompt pass/fail, latency, and validation errors
- final batch summary

If the process is interrupted, it records an `interrupted` run state and keeps the partial artifacts.

## GPU Notes

- GPU use is never automatic in this sandbox.
- The runner uses CPU-only mode unless you explicitly pass `--n-gpu-layers` greater than `0`.
- Even with `--n-gpu-layers`, GPU offload only works when the installed `llama.cpp` backend was built with GPU support and the current environment can actually see the Nvidia runtime.
- The runner now reports this at startup so there is no ambiguity about whether the run is CPU-only or attempting GPU offload.

## Docker Sandbox GPU Prerequisite

If you launch the workspace with the default `docker sandbox run opencode` flow, do not assume the container has CUDA access.

Before restarting Experiment 1 in a GPU-backed container, verify from the host shell that Docker can see the Nvidia device at all:

```powershell
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

If that command fails, the problem is outside the Python experiment. The Docker daemon is not exposing an Nvidia GPU to Linux containers yet, so Experiment 1 will be CPU-only no matter what flags the runner uses.

If the Docker test succeeds, recreate the opencode sandbox with GPU access enabled. The default container that was previously used for this project was started without any GPU device request, so it could never see the RTX 4060.

## What The Runner Proves

- The GBNF grammar constrains structure, exact keys, enum values, and bans extra prose.
- The Python validator checks required keys, key order, numeric finiteness, and numeric ranges.
- Results are split into:
  - `syntax_pass`
  - `schema_pass`
  - `range_pass`
  - `overall_pass`

## Important Note About Numeric Ranges

The grammar is intentionally strict about JSON structure but does not try to encode every float range directly in GBNF. That keeps decoding tractable. Numeric bounds are enforced immediately after parsing against the local schema contract.

## Suggested First Benchmark Loop

1. Run the fixture set with temperature `0.0`.
2. Inspect the pass rates and latency distribution.
3. If semantics are weak but syntax is perfect, refine the system prompt or add a few-shot variant.
4. If latency is too high, try a smaller quantization or model.

## Expected Outputs

- `results/run_<timestamp>.json`
- `logs/latency_<timestamp>.csv`
- `logs/status_<timestamp>.log`

These artifacts are designed to feed directly into the project wiki and later C++ integration decisions.
