# Experiment 1.1: The Advanced Brain Sandbox

This experiment extends `Experiment 1` in three ways:

- a larger nested synth parameter surface
- a forced leading `explanation` key for visible reasoning
- sparse patch output using `null` for no-op parameters

## Goal

- prove that a local LLM can emit one strict nested JSON patch object
- force a visible `explanation` string as the first key
- prove sparse updates by requiring unchanged parameters to stay `null`
- measure the latency penalty of a larger schema compared with Experiment 1

## Files

- `brain_sandbox_advanced.py` - advanced nested sparse runner
- `prompts/system_prompt_v1_1.txt` - advanced sparse system prompt
- `grammars/advanced_patch_v1.gbnf` - explanation-first nested grammar
- `schemas/advanced_patch_v1.json` - canonical sparse nested contract
- `fixtures/baseline_prompts.json` - baseline full and sparse prompts
- `fixtures/sparse_prompts.json` - targeted sparse-update prompts
- `fixtures/adversarial_prompts.json` - adversarial prompts for formatting and range pressure
- `results/` - timestamped batch JSON artifacts
- `logs/` - timestamped status and latency artifacts

## Runtime

This experiment is designed to reuse the existing `sound-forge-brain-sandbox:gpu` image from Experiment 1.

You only need to rebuild if:

- `Experiments/Experiment 1/Dockerfile.gpu` changed
- Python dependencies changed

## One-line host run

```powershell
& ".\Experiments\Experiment 1.1\run_experiment_gpu.ps1" -ModelPath "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" -GpuLayers 35 -MaxTokens 640
```

## Watch progress

```powershell
Get-Content ".\Experiments\Experiment 1.1\logs\status_latest.log" -Wait
```

## Artifacts

- `Experiments\Experiment 1.1\logs\status_latest.log`
- `Experiments\Experiment 1.1\logs\latency_<timestamp>.csv`
- `Experiments\Experiment 1.1\results\run_<timestamp>.json`

## Current Status

- default generation budget is now `MaxTokens 640`; the initial `320` token limit truncated full nested outputs
- authoritative baseline runs:
  - `results/run_20260504T045814Z.json` - Qwen 2.5 3B baseline (`2/5`, sparse failures on context-aware prompts)
  - `results/run_20260504T045919Z.json` - Qwen 2.5 7B baseline (`5/5`)
- authoritative adversarial runs:
  - `results/run_20260504T050047Z.json` - Qwen 2.5 3B adversarial (`3/5`)
  - `results/run_20260504T050156Z.json` - Qwen 2.5 7B adversarial (`4/5`, numeric-pressure failure only)
- advisor-facing report:
  - `reports/experiment-1-1-advanced-brain-sandbox-report.md`

## Metrics of interest

- model load time
- mean and p95 inference latency
- parse/validation/sparse extraction latency
- explanation length
- changed parameter count
- null ratio
- all-null no-op count
