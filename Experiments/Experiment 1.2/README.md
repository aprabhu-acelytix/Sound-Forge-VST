# Experiment 1.2: Patch-Only Async Pipeline Sandbox

This experiment extends `Experiment 1.1` into a patch-first control path designed to answer three questions before the eventual JUCE/C++ handoff:

- does removing explanation generation from the critical path materially reduce applied-patch latency
- does compact delta output outperform full sparse-null output enough to justify the extra downstream normalization step
- can clamp-aware middleware safely preserve usability under numeric-pressure prompts without hiding raw model weaknesses

## Goals

- separate machine control latency from human explanation latency
- compare `sparse_patch_only` against `compact_delta`
- compare raw range obedience against post-parse clamping
- support true chat-turn few-shot examples for 3B sparse-delta evaluation
- simulate an async `Patch -> Apply -> Explanation` worker pipeline with stale-explanation protection

## Files

- `brain_sandbox_experiment_1_2.py` - core async runner with clamp middleware and metrics
- `summarize_results.py` - compares one or more Experiment 1.2 result artifacts
- `run_experiment_gpu.ps1` - host wrapper for Docker execution
- `run_benchmark_matrix.ps1` - seeded benchmark matrix for Tests A, B, and C
- `configs/parameter_registry_v1.json` - canonical parameter metadata and clamp policy registry
- `schemas/sparse_patch_only_v1.json` - full sparse patch-only contract
- `schemas/compact_delta_patch_v1.json` - compact delta contract
- `schemas/explanation_response_v1.json` - explanation worker output contract
- `grammars/sparse_patch_only_v1.gbnf` - grammar for full sparse patch-only output
- `grammars/compact_delta_patch_v1.gbnf` - grammar for compact delta output
- `grammars/explanation_response_v1.gbnf` - grammar for explanation worker output
- `fixtures/test_a_latency_prompts.json` - seed latency corpus for two-pass and compact-delta comparisons
- `fixtures/test_b_numeric_pressure_prompts.json` - seed numeric-pressure corpus for clamp evaluation
- `fixtures/test_c_sparse_ground_truth_prompts.json` - seed sparse ground-truth corpus for few-shot evaluation
- `fixtures/fewshot_examples_seed.json` - seed user/assistant few-shot examples for 3B testing
- `results/` - run JSON artifacts
- `logs/` - status logs, latency CSV files, and clamp event logs

## Runtime

This experiment reuses the existing `sound-forge-brain-sandbox:gpu` Docker image from `Experiment 1.1`.

You only need to rebuild if:

- `Experiments/Experiment 1/Dockerfile.gpu` changed
- Python dependencies changed

## One-line host run

```powershell
& ".\Experiments\Experiment 1.2\run_experiment_gpu.ps1" `
    -PatchModelPath "C:\LLMs\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" `
    -PatchContract sparse_patch_only `
    -ExplanationMode llm `
    -ExplainerModelPath "C:\LLMs\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf" `
    -PromptsFile "Experiments/Experiment 1.2/fixtures/test_a_latency_prompts.json"
```

## Watch progress

```powershell
Get-Content ".\Experiments\Experiment 1.2\logs\status_latest.log" -Wait
```

## Seed Fixture Fields

Every prompt case supports these shared fields:

- `id`
- `prompt`
- `tags`
- `stream_id`
- `current_patch_context`
- `expected_changed_paths`
- `expected_exact_values`
- `expected_null_paths`
- `expected_clamped_paths`
- `expected_noop`

Intended usage:

- Test A latency corpus focuses on `patch_applied_ms`, `explanation_ready_ms`, and `ui_gap_ms`
- Test B numeric-pressure corpus focuses on `expected_clamped_paths`
- Test C sparse ground-truth corpus focuses on `expected_changed_paths`, `expected_exact_values`, and `expected_null_paths`

## Key Metrics

- `patch_ready_ms` - patch generation plus parse plus validation completion
- `patch_applied_ms` - patch clamped, merged, and safely applied in the mock APVTS layer
- `explanation_ready_ms` - explanation available for UI display
- `ui_gap_ms` - explanation lag after the sound was already applied
- `raw_range_pass_rate` - model obeyed numeric bounds without middleware help
- `clamp_intervention_rate` - middleware had to step in
- `clamp_expectation_pass_rate` - expected clamp paths matched observed clamp paths
- `sparse_exact_match_rate` - changed-path behavior matched the corpus expectation

## Suggested Runs

- Test A1: `sparse_patch_only` + `explanation off`
- Test A2: `sparse_patch_only` + async `llm` explanation
- Test A3: `compact_delta` + `explanation off`
- Test B1: Qwen 7B numeric-pressure clamp evaluation
- Test B2: Mistral 7B numeric-pressure clamp evaluation
- Test C0: Qwen 3B compact or sparse zero-shot
- Test C1: Qwen 3B with 2-shot chat-turn few-shot examples

## Notes

- explanations always describe the final sanitized patch, never raw model intent
- stale explanations are cancelled or discarded by `stream_id` and `patch_version`
- raw model correctness and post-clamp safety remain separate metrics by design
