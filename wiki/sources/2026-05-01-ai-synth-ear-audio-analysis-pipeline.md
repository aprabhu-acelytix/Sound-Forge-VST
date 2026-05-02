# AI Synth Ear Audio Analysis Pipeline

## Summary

This source designs the Ear as a layered local analysis pipeline for the synthesizer. It recommends deterministic DSP descriptors as the always-on base, lightweight neural tagging for human-readable semantics, optional embedding-based texture analysis for richer retrieval, and a strict queue-based separation between real-time capture, background analysis, and Brain-facing transport.

## Key Points

- The Ear should convert live synth audio into structured context for the Brain rather than exposing raw waveform buffers.
- Essentia streaming is a strong fast path for descriptors like MFCCs, spectral centroid, zero-crossing rate, and onset features.
- Audio should move from `processBlock` to background analysis through a lock-free SPSC queue.
- `sherpa-onnx` Zipformer tagging is a promising lightweight semantic layer on top of deterministic features.
- CLAP and TRR-style embeddings can provide open-vocabulary or texture-aware context, but they are higher-cost branches.
- Ear outputs should be aggregated into JSON or a similar compact payload and handed to the Brain through a separate queue rather than by performing network work inline.
- Ear inference should stay local whenever possible because remote audio analysis quickly becomes too stale for responsive control.

## Claims

- The Ear works best as a layered system rather than a single model.
- Deterministic DSP descriptors remain valuable even when neural tagging is available.
- A local background analysis path is necessary to balance semantic richness with plugin latency constraints.
- TRR-style texture representations are better suited than simple mean-pooled embeddings when the project needs fine-grained timbral texture matching.
- ONNX deployment and Essentia licensing are architectural decisions, not just implementation details.

## Connections

- [[wiki/concepts/ear-audio-analysis-pipeline]] - Distills the source into the vault's Ear design concept.
- [[wiki/concepts/ear-brain-tutor-architecture]] - Places the Ear inside the broader System of Models.
- [[wiki/concepts/realtime-safe-ai-audio-architecture]] - Captures the thread and queue boundaries required by the source.
- [[wiki/entities/essentia]] - The main deterministic fast-path analysis library.
- [[wiki/entities/sherpa-onnx]] - The proposed lightweight semantic tagging layer.
- [[wiki/entities/onnx-runtime]] - The runtime for the neural analysis branches.
- [[wiki/overview/ai-vst3-synthesizer-architecture]] - Updates the top-level architecture with a more detailed Ear design.
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]] - Refines the current build recommendation.

## Open Questions

- What is the smallest Ear implementation that gives the Brain meaningfully better control decisions?
- Should Essentia be used directly given its AGPL licensing, or should the fast path be reimplemented with a different stack?
- Are CLAP or TRR worth the added model complexity before the basic Ear pipeline is proven?

## Source

- Raw file: [[raw/sources/AI Synth Ear_ Audio Analysis Pipeline]]
