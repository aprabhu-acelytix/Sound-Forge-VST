# Ear Audio Analysis Pipeline

## What It Is

The Ear audio analysis pipeline converts the synthesizer's live output into structured acoustic context that the Brain can use. It captures audio safely from the real-time thread, analyzes it on background workers, and emits compact descriptors, semantic tags, and optional texture-aware embeddings.

## Why It Matters

The Brain cannot reason well from raw waveform buffers. The Ear is the bridge from real-time DSP to semantic control. A layered design gives the system fast deterministic features first, then richer meaning only where the latency and CPU budget allow it.

## Key Details

- The audio thread should push samples into a preallocated SPSC FIFO and do nothing more than bounded capture work.
- A background analysis thread should drain larger windows from the FIFO and run the Ear's heavy work off-thread.
- The fast path should use Essentia streaming primitives such as `RingBufferInput`, `FrameCutter`, `Windowing`, and `Spectrum` to compute descriptors like MFCCs, spectral centroid, zero-crossing rate, and onset features.
- A slower semantic branch can classify larger windows through ONNX-based audio tagging, such as `sherpa-onnx` Zipformer models, to produce human-readable labels with confidence scores.
- Optional advanced branches such as CLAP or TRR-style texture embeddings should be treated as higher-cost additions for open-vocabulary or texture-sensitive retrieval rather than a first requirement.
- Ear outputs should be aggregated into a compact JSON or equivalent structured payload and handed to the Brain through a separate application or network queue, not by making HTTP calls inline from the analysis worker.
- Local Ear inference is preferred because remote audio analysis quickly makes the feedback loop stale.

## Related Pages

- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/entities/essentia]]
- [[wiki/entities/sherpa-onnx]]
- [[wiki/entities/onnx-runtime]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]

## Open Questions

- What exact Ear output schema should the Brain consume first?
- Should v1 stop at Essentia plus semantic tags, or include texture embeddings as well?
- Is Essentia's AGPL licensing acceptable for the intended distribution model?
