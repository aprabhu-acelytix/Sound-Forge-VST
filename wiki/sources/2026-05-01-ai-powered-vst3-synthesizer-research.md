# AI-Powered VST3 Synthesizer Research

## Summary

This source argues that an agentic VST3 synthesizer is feasible only with strict separation between real-time DSP and AI workloads. It recommends an Ear-Brain-Tutor architecture, lock-free cross-thread handoff, constrained parameter payloads, and a hybrid synthesis engine that pairs anti-aliased wavetable generation with physical modeling.

## Key Points

- The audio thread must avoid blocking work, heap allocation, locks, file I/O, and unbounded inference.
- The Ear should combine fast algorithmic analysis with slower semantic tagging or retrieval.
- The Brain should emit strict JSON parameter payloads rather than conversational text.
- APVTS writes belong on the message thread, while the audio thread reads atomic parameter values.
- Out-of-process LLM serving is the safest default for heavyweight models inside a DAW environment.
- The synth engine should combine wavetable flexibility with physically modeled expressiveness.

## Claims

- Heavy model inference and HTTP calls must be isolated from `processBlock`.
- An SPSC queue plus message-thread APVTS updates is the safest core control-flow pattern.
- Essentia is a strong fast-path Ear, while ONNX-based models belong on background threads.
- `llama.cpp` is viable for embedded local inference, but a local API server such as Ollama is the safer default for stability and memory isolation.
- DaisySP is a strong foundation for the physical-modeling side of the synthesizer, while RTNeural is reserved for tiny inline neural DSP blocks.

## Connections

- [[wiki/overview/ai-vst3-synthesizer-architecture]] - Distills the source into the current top-level system thesis.
- [[wiki/concepts/ear-brain-tutor-architecture]] - Captures the proposed System of Models split.
- [[wiki/concepts/realtime-safe-ai-audio-architecture]] - Extracts the non-negotiable thread and synchronization rules.
- [[wiki/concepts/structured-parameter-mapping]] - Tracks how the Brain turns language into safe control data.
- [[wiki/concepts/hybrid-synthesis-engine]] - Captures the recommended DSP backend strategy.
- [[wiki/entities/juce]] - The host framework for the plugin and thread boundaries.
- [[wiki/entities/ollama]] and [[wiki/entities/llama-cpp]] - The main Brain deployment options.
- [[wiki/entities/essentia]], [[wiki/entities/onnx-runtime]], [[wiki/entities/daisysp]], and [[wiki/entities/rtneural]] - The main supporting libraries.

## Open Questions

- Should the first Brain deployment be embedded `llama.cpp` or an external local model server?
- Which slow-path Ear capability should come first: audio tagging, retrieval, or captioning?
- What is the smallest synth parameter schema that still makes conversational control meaningful?

## Source

- Raw file: [[raw/sources/AI-Powered VST3 Synthesizer Research]]
