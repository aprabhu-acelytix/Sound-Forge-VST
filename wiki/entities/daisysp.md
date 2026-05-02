# DaisySP

## Overview

DaisySP is a lightweight C++ DSP library originally developed for embedded audio hardware and known for efficient physical modeling, resonators, and modular-style synthesis blocks.

## Relevance

The sources recommend DaisySP as the strongest starting point for the physical-modeling side of the hybrid engine. Its efficiency and breadth make it well suited to strings, resonators, percussion, and other expressive sound-generation modules.

Useful implementation details from the current sources:

- DaisySP's sample-based `Process()` style is a good fit when the engine needs manual control over routing and single-sample feedback structures.
- It is viable on desktop JUCE projects through standard CMake integration rather than only on Daisy hardware.
- It becomes most useful when wrapped in a custom graph or engine layer rather than treated as a complete plugin architecture by itself.
- Its per-sample primitives leave room for custom fractional-delay tuning, saturation, DC blocking, and other safety logic around feedback-heavy models.
- Desktop builds may need guards around hardware-specific optimizations or inline assembly that were originally targeted at embedded environments.

## Related Concepts

- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/modulation-matrix-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Which DaisySP modules are the best fit for a first playable synth architecture?
- Which DaisySP primitives should live inside the first custom sample-based engine graph?
