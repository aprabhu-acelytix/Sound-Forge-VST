# AI Inference in JUCE Plugins

## Summary

This source focuses on the concrete engineering required to deploy local AI runtimes inside or alongside a JUCE plugin. It distinguishes between embedded `llama.cpp`, out-of-process local model servers, background-only ONNX Runtime sessions, and inline RTNeural models, arguing that each belongs to a different execution domain with different build, memory, and failure-isolation tradeoffs.

## Key Points

- Embedded `llama.cpp` requires careful CMake integration, explicit model and context lifetime management, and background-thread-only decode loops.
- Local model servers such as Ollama should be treated as streaming NDJSON endpoints and called through robust background HTTP clients.
- `juce::WebInputStream` is presented as a weak fit for long-lived streamed inference responses compared with lower-level HTTP libraries.
- ONNX Runtime must be constrained aggressively inside plugins through thread limits and spinning controls.
- RTNeural is the right class of tool for tiny inline neural DSP, not for heavyweight inference.
- Build and packaging strategy are part of the runtime architecture because they affect binary size, memory behavior, and host crash boundaries.

## Claims

- Heavy AI inference can coexist with JUCE plugins only if each runtime is assigned an appropriate execution domain.
- External model serving is the safest default for heavyweight language models in a DAW environment.
- Embedded `llama.cpp` is viable, but only with explicit background-thread isolation and careful resource budgeting.
- ORT defaults are too aggressive for plugin use and must be tuned down.
- RTNeural succeeds because it matches the audio thread's zero-allocation, deterministic execution model.

## Connections

- [[wiki/overview/ai-vst3-synthesizer-architecture]] - Refines the vault's top-level deployment recommendations.
- [[wiki/concepts/realtime-safe-ai-audio-architecture]] - Extends the execution-domain model for different inference runtimes.
- [[wiki/entities/juce]] - Adds JUCE-specific integration and callback details.
- [[wiki/entities/llama-cpp]] - Adds concrete embedded-runtime guidance.
- [[wiki/entities/ollama]] - Adds streaming local API design details.
- [[wiki/entities/onnx-runtime]] - Adds constrained session and packaging guidance.
- [[wiki/entities/rtneural]] - Adds inline neural DSP implementation details.
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]] - Updates the current recommendation with deployment specifics.

## Open Questions

- Should the first Brain implementation be exclusively external, or should the project prototype embedded `llama.cpp` early as well?
- Which HTTP client should become the standard for local model streaming?
- When does ONNX become worth the extra packaging complexity versus simpler algorithmic analysis?

## Source

- Raw file: [[raw/sources/AI Inference in JUCE Plugins]]
