# Experiment 1.2 Report

## Summary

Experiment 1.2 evaluated whether patch-first generation, clamp-aware middleware, compact delta output, and few-shot prompting improve the Sound Forge Brain control path enough to justify the move toward the eventual JUCE handoff.

## Test A: Latency And Human Overhang

Variant | Patch Applied | Explanation Ready | UI Gap | Human Overhang | Patch Pass
--- | --- | --- | --- | --- | ---
A0 | 10626.85 ms | 10626.85 ms | 0.00 ms | 0.00x | 100.0%
A1 | 52015.82 ms | n/a | n/a | n/a | 100.0%
A2 | 54217.17 ms | 56386.29 ms | 2169.12 ms | 0.04x | 100.0%
A3 | 27234.77 ms | n/a | n/a | n/a | 100.0%

## Test B: Numeric Pressure And Clamping

Variant | Raw Range Pass | Clamp Intervention | Clamp Expectation Pass
--- | --- | --- | ---
B1 | 14.3% | 85.7% | 57.1%
B2 | 0.0% | 0.0% | n/a

## Test C: Sparse Salvage

Variant | Sparse Exact Match | Patch Pass
--- | --- | ---
C0 | 75.0% | 100.0%
C1 | 75.0% | 100.0%
C2 | 75.0% | 100.0%

## Interpretation

- Test A human overhang: A0=0.00x, A1=n/a, A2=0.04x, A3=n/a.
- Test B clamp comparison: Qwen 7B clamp intervention=85.7%, Mistral 7B clamp intervention=0.0%.
- Test C sparse salvage: zero-shot=75.0%, 2-shot=75.0%, 3-shot=75.0%.

## Notes

- A0 is normalized from the Experiment 1.1 explanation-first contract. Its UI gap is treated as zero because machine application cannot occur until the full explanation-first payload exists.
- Sparse exact match is still the decisive metric for Test C because safe apply alone can hide raw model delta failures.
