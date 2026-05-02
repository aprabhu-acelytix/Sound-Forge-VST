# Modulation Matrix Architecture

## What It Is

The modulation matrix is the control-routing layer that connects sources such as LFOs, envelopes, velocity, and AI-generated macro signals to synthesis parameters. In this project it must handle large numbers of routings efficiently and allow the AI to change topology without corrupting the real-time engine.

## Why It Matters

An AI-driven synth is not just selecting preset values. It is effectively patching a moving network of modulation relationships. If the matrix is cache-unfriendly, branch-heavy, or mutated unsafely, the engine will lose performance or stability before the DSP algorithms themselves become the bottleneck.

## Key Details

- Favor Structure of Arrays over Array of Structures so the CPU reads contiguous source, destination, and depth data instead of cache-polluting objects.
- Pre-calculate modulation blocks before the core synthesis pass so oscillators and resonators can read ready-made control data instead of evaluating routing logic repeatedly.
- Favor fused SIMD-style loops over repeated passes with generic vector helpers when the routing count is high enough to become memory-bound.
- Inside JUCE, this points toward fused `juce::dsp::SIMDRegister` loops rather than chaining multiple `juce::FloatVectorOperations` passes once memory traffic becomes the bottleneck.
- Align and pad modulation buffers so SIMD lanes can run cleanly without scalar cleanup dominating the hot path.
- Build larger topology changes on a background thread and switch the active matrix through an atomic pointer or dual-buffer exchange rather than mutating the live matrix in place.
- Retire old topology objects on the worker side after the swap so the audio callback only pays for the atomic exchange.
- Treat the modulation matrix as part of the DSP architecture because routing density, data layout, and swap semantics directly affect voice count and glitch resistance.

## Related Pages

- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/entities/juce]]
- [[wiki/entities/daisysp]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]

## Open Questions

- Which modulation sources should be global, per-voice, or per-note in the first implementation?
- How many simultaneous routings per voice should the initial engine target?
- Should matrix topology swaps occur at block boundaries only, or on another explicitly defined safe point?
