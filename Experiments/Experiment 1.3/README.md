# Experiment 1.3: The Convergence Sandbox

This experiment extends `Experiment 1.2` to resolve the remaining architecture questions before any C++ move:

- combine explanation-first reasoning with compact-delta output in a single contract
- debug model-specific prompt serialization and grammar compatibility, especially for Mistral 7B
- re-run sparse fidelity evaluation on `compact_delta` for the Qwen 3B model

## Phase 1

Phase 1 focuses on contracts and compatibility only.

- canonical no-leading-whitespace grammars
- hybrid `explanation + compact_delta` contract
- compact-delta-specific few-shot examples for no-ops and same-value skip behavior
- explicit per-model `chat_format` routing
- smoke-test prompt-token sanity checks for Mistral

## Files

- `brain_sandbox_experiment_1_3.py` - Experiment 1.3 runner with prompt-token sanity checks
- `run_experiment_gpu.ps1` - host wrapper for GPU execution with per-model chat-format routing
- `schemas/compact_delta_patch_v2.json` - canonical compact-delta contract
- `schemas/hybrid_explanation_delta_v1.json` - explanation-first + compact-delta hybrid contract
- `grammars/compact_delta_patch_v2.gbnf` - canonical no-leading-whitespace compact-delta grammar
- `grammars/hybrid_explanation_delta_v1.gbnf` - canonical no-leading-whitespace hybrid grammar
- `fixtures/fewshot_examples_compact_delta.json` - compact-delta user/assistant few-shot examples

## Phase 1 Smoke Tests

- `M0`: Mistral one-prompt numeric-pressure smoke test with grammar disabled
- `M1`: Mistral one-prompt numeric-pressure smoke test with canonical compact-delta grammar enabled

Both smoke tests should confirm that prompt-token counts are in the expected full-prompt range before any full matrix run.
