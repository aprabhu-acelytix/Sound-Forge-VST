# Experiment 1.2 Report

## Summary

Experiment 1.2 evaluated whether patch-first generation, clamp-aware middleware, compact delta output, and few-shot prompting improve the Sound Forge Brain control path enough to justify the move toward the eventual JUCE handoff.

## Test A: Latency And Human Overhang

Variant | Patch Applied | Explanation Ready | UI Gap | Human Overhang | Patch Pass
--- | --- | --- | --- | --- | ---
A0 | 10608.78 ms | 10608.78 ms | 0.00 ms | 0.00x | 100.0%
A1 | 52750.75 ms | n/a | n/a | n/a | 100.0%
A2 | 55364.35 ms | 57541.89 ms | 2177.54 ms | 0.04x | 100.0%
A3 | 21071.94 ms | n/a | n/a | n/a | 75.0%

## Test B: Numeric Pressure And Clamping

Variant | Raw Range Pass | Clamp Intervention | Clamp Expectation Pass
--- | --- | --- | ---
B1 | 14.3% | 85.7% | 57.1%

## Test C: Sparse Salvage

Variant | Sparse Exact Match | Patch Pass
--- | --- | ---

## Interpretation

- Test A human overhang: A0=0.00x, A1=n/a, A2=0.04x, A3=n/a.
- Test B clamp comparison: Qwen 7B clamp intervention=85.7%, Mistral 7B clamp intervention=n/a.
- Test C sparse salvage: zero-shot=n/a, 2-shot=n/a, 3-shot=n/a.

## Notes

- A0 is normalized from the Experiment 1.1 explanation-first contract. Its UI gap is treated as zero because machine application cannot occur until the full explanation-first payload exists.
- Sparse exact match is still the decisive metric for Test C because safe apply alone can hide raw model delta failures.
