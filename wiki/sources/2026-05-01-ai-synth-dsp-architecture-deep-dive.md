# AI Synth DSP Architecture Deep Dive

## Summary

This source focuses on the core DSP engine beneath the AI system. It argues for a hybrid engine built from anti-aliased wavetable synthesis, sample-based physical modeling primitives, explicit feedback safety rails, and a cache-friendly modulation matrix that the AI can reconfigure safely at runtime.

## Key Points

- The engine should combine a band-limited wavetable core with physical-model components rather than relying on a single synthesis family.
- DaisySP-style sample-based primitives are a better fit than rigid or block-oriented graph layers when the AI needs runtime rewiring and feedback.
- Wavetable quality depends on FFT-based mipmapping and high-quality interpolation, not just storing a table.
- Smooth wavetable pitch sweeps need adjacent mipmap crossfades, and higher mip levels should preserve harmonic amplitudes instead of being renormalized.
- Physical-model loops need explicit stability protections against denormals, runaway feedback, DC buildup, and NaN or Inf propagation.
- Karplus-Strong and similar waveguide structures should use all-pass fractional delay when accurate tuning and brightness matter.
- The modulation matrix should use SoA-style layout and SIMD-friendly fused loops when routing density becomes large.
- In JUCE-based SIMD code, fused `juce::dsp::SIMDRegister` loops are preferred over repeated helper passes once modulation becomes memory-bound.
- Larger graph or matrix topology changes should be assembled off-thread and swapped atomically rather than edited in place.

## Claims

- DSP graph flexibility is a first-order architectural concern for an AI-driven synth, not an implementation detail.
- `juce::AudioProcessorGraph` and compile-time-only chains are poor fits for feedback-rich, AI-rewired synthesis topologies.
- A synth that tolerates aggressive AI modulation needs both high-quality anti-aliasing and hard safety rails in recursive structures.
- Modulation data layout can become as important as the underlying synthesis algorithms once routing complexity grows.

## Connections

- [[wiki/concepts/hybrid-synthesis-engine]] - Captures the hybrid wavetable and physical-model strategy.
- [[wiki/concepts/modulation-matrix-architecture]] - Captures the routing and data-layout implications.
- [[wiki/concepts/realtime-safe-ai-audio-architecture]] - Connects the engine swaps to lock-free real-time rules.
- [[wiki/entities/daisysp]] - The strongest library match for the sample-based physical-model side.
- [[wiki/entities/juce]] - Clarifies where JUCE helps and where custom graph code is preferable.
- [[wiki/overview/ai-vst3-synthesizer-architecture]] - Sharpens the overall engine thesis.
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]] - Refines the staged implementation order.

## Open Questions

- Which specific wavetable oscillator and physical-model modules belong in the first playable engine?
- How much of the engine should be fully custom versus wrapped around DaisySP and JUCE utility blocks?
- What is the first acceptable target for modulation-routing density per voice?

## Source

- Raw file: [[raw/sources/AI Synth DSP Architecture Deep Dive]]
