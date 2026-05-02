# sherpa-onnx

## Overview

`sherpa-onnx` is a C and C++ deployment framework built around ONNX Runtime for speech and audio models. In this vault it is most relevant as a lightweight way to run pretrained audio-tagging models for the Ear.

## Relevance

The Ear source positions `sherpa-onnx` as a strong candidate for semantic audio tagging because it wraps ONNX deployment cleanly and supports compact Zipformer-based tagging models. It is useful when the project needs human-readable labels such as timbral or event tags on top of deterministic DSP descriptors.

Useful implementation details from the current sources:

- Audio-tagging models can run on background threads over larger windows such as around one second of audio.
- Quantized Zipformer checkpoints keep model size relatively modest, making plugin-side packaging more realistic than many larger classifiers.
- The output is a label-and-confidence layer that complements algorithmic descriptors rather than replacing them.
- It still inherits ONNX Runtime's threading and deployment constraints, so it belongs off the audio thread.

## Related Concepts

- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]

## Open Questions

- Are the available pretrained labels aligned well enough with synthesizer timbres, or will the project need a more synth-specific tagging vocabulary?
