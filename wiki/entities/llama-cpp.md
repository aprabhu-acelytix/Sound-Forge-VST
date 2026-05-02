# llama.cpp

## Overview

`llama.cpp` is a native C and C++ inference engine for local large language models, typically loaded from GGUF weights and accelerated through CPU or GPU backends.

## Relevance

The sources treat `llama.cpp` as the main option for embedding the Brain directly inside the plugin ecosystem. Its strengths are local execution, quantized model support, and grammar-constrained decoding. Its cost is significant CPU, memory, binary-size, and crash-coupling pressure, so it belongs on background threads only.

Useful implementation details from the current sources:

- In JUCE projects it is typically integrated as a static CMake dependency with non-essential examples, tests, and server targets disabled.
- GGUF model weights should usually be memory-mapped when possible, while `llama_model` and `llama_context` lifetimes stay clearly separated.
- `llama_decode()` must run on lowered-priority worker threads, not on the message or audio thread.
- CPU thread counts should usually be sized around physical cores rather than all logical threads.
- Embedded deployment improves self-containment, but every plugin instance pays the memory and failure-isolation cost.
- GBNF grammars are one of its strongest advantages in this project because they can physically constrain output structure and allowed keys at token-selection time.
- Grammar design should stay bounded and specific; overly permissive or deeply branching grammars can hurt inference performance.

## Related Concepts

- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]]
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Is embedded inference worth the per-instance memory cost and host crash coupling compared with a shared local model server?
- How strict should the grammar be in v1: full parameter object generation, or smaller bounded sub-schemas per task?
