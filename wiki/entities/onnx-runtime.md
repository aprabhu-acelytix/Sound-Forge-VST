# ONNX Runtime

## Overview

ONNX Runtime is a general-purpose inference engine for models exported from training frameworks such as PyTorch and TensorFlow.

## Relevance

The sources recommend ONNX Runtime for the Ear's background-only neural analysis tasks, such as audio tagging and lightweight classifiers. It is explicitly not treated as real-time safe because of heap allocation and internal thread-pool behavior.

Useful implementation details from the current sources:

- In plugin projects it is usually integrated as a dynamically linked prebuilt runtime or a pruned custom build rather than built from source in-tree.
- Sessions should usually clamp `SetIntraOpNumThreads(1)` and `SetInterOpNumThreads(1)` for plugin safety.
- Thread spinning should be disabled through session config entries to avoid background CPU starvation.
- Graph optimizations, pruning, and quantization matter because AVX-heavy inference can throttle clocks and evict DSP working sets from cache.
- Ear use cases include lightweight audio tagging, CLAP-style embedding inference, and texture-oriented feature extraction, all from background threads only.
- Dynamic linking is often preferred not just for size but to reduce symbol and protobuf conflicts between plugins loaded into the same host.

## Related Concepts

- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/ear-brain-tutor-architecture]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Which execution providers and thread settings give the best balance of speed, power draw, and DAW stability?
- How should pruned or dynamic ORT binaries be packaged to avoid deployment friction across hosts?
- Which Ear models justify ONNX complexity in v1, and which can wait until after the deterministic fast path is stable?
