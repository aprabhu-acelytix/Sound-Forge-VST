# Ear Brain Tutor Architecture

## What It Is

This is the agentic split proposed for the synthesizer: the Ear observes the sound, the Brain turns language plus sonic context into parameter decisions, and the Tutor exposes that system to the user through a conversational interface.

## Why It Matters

The separation gives each part of the system a clear job and keeps the AI layers from collapsing into the real-time DSP path. It also makes the architecture easier to reason about, test, and swap out over time.

## Key Details

- The Ear should combine a continuous algorithmic fast path with slower neural or embedding branches when richer context is needed.
- The Ear should emit compact numeric and semantic summaries or structured JSON rather than raw waveform dumps.
- The Brain emits constrained parameter payloads rather than free-form prose.
- The Tutor lives on the UI side and orchestrates user requests, explanations, and triggered analysis.
- The Ear should stay local enough that the Brain reasons about current audio rather than stale remote analysis.
- The source treats this as a System of Models rather than a single monolithic model.

## Related Pages

- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/entities/essentia]]
- [[wiki/entities/sherpa-onnx]]
- [[wiki/entities/onnx-runtime]]
- [[wiki/entities/llama-cpp]]
- [[wiki/entities/ollama]]
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- What is the smallest useful Tutor feature for a first prototype?
- Should the first Ear implementation prioritize algorithmic descriptors, semantic tags, or both?
- Should advanced embedding methods such as CLAP or TRR wait until the fast path and tagger are stable?
