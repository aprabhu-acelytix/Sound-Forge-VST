# Hybrid Synthesis Engine

## What It Is

The source material recommends a hybrid engine that combines anti-aliased wavetable synthesis with physical modeling. The goal is to cover both precise, programmable timbres and more organic, unstable acoustic behaviors, while still allowing the AI to reconfigure the engine without destabilizing the audio thread.

## Why It Matters

This pairing gives the agentic system a richer parameter space to control. Wavetable synthesis supports broad timbral coverage, while physical modeling introduces expressive resonances and non-linear behavior that would be difficult to reach with subtractive controls alone.

## Key Details

- Use FFT-based mipmapped or otherwise band-limited wavetable generation to prevent aliasing at high pitches and under extreme modulation; zero DC during table generation and avoid re-normalizing higher mip levels so timbre stays consistent across the keyboard.
- Crossfade adjacent mip levels during pitch sweeps so table changes do not click when the oscillator crosses a band boundary.
- Use high-quality interpolation, such as cubic Hermite or a comparable approach, rather than naive table lookup or simple linear interpolation.
- Use JUCE DSP for filters, oversampling, and utility blocks, but avoid relying on `juce::AudioProcessorGraph` when the engine needs sample-level feedback or AI-directed graph rewiring.
- Use DaisySP or similar sample-based primitives for strings, resonators, percussion, and other physical-model components that benefit from manual graph control.
- Physical models require strict parameter bounds plus runtime safety rails such as denormal mitigation, soft saturation, DC blocking, and NaN or Inf guards.
- Karplus-Strong and related waveguide structures should prefer first-order all-pass fractional delay over simple linear interpolation when they need to stay bright and accurately tuned.
- Modal synthesis is a strong fit for metallic, stiff, or percussive bodies where simple waveguides are too limited, and exposes useful AI-facing controls such as mode distribution, damping, and inharmonicity.
- A high-throughput modulation matrix is part of the engine architecture, not a separate afterthought.
- RTNeural may be appropriate for tiny neural signal-path modules, but not for large generative or classification models.

## Related Pages

- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/concepts/modulation-matrix-architecture]]
- [[wiki/entities/juce]]
- [[wiki/entities/daisysp]]
- [[wiki/entities/rtneural]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Which wavetable and physical-model modules should be included in the first engine version?
- Should the first prototype emphasize preset generation, live conversational control, or adaptive listening?
- Should the first engine graph be fully custom and sample-based from the start, or should JUCE utility layers wrap a smaller custom core?
