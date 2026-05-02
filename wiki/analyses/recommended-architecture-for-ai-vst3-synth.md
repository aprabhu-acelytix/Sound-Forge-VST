# Recommended Architecture For AI VST3 Synth

## Question

What is the current recommended architecture for building an AI-powered VST3 synthesizer from the available research?

## Answer

Use JUCE as the plugin framework and keep the audio thread limited to DSP, preallocated FIFO writes, denormal-protected math, and lock-free state reads. Run the Ear and Brain on background workers, prefer Essentia for continuous low-cost timbre features, add slower ONNX-based semantic analysis only on demand, and use `juce::AbstractFifo` plus acquire-release atomics for bounded cross-thread handoff. For host-visible controls, route parsed parameter payloads onto the message thread through a bounded queue and apply them there through APVTS change gestures. `juce::AsyncUpdater` remains viable, but timer-based message-thread polling is also a strong default when the project wants to avoid excessive async wakeups. For internal DSP-only macro state, prefer direct POD payload transfer without forcing everything through APVTS.

For the core synth engine, prefer a custom sample-based graph around low-level DSP primitives rather than leaning on a fully dynamic `juce::AudioProcessorGraph`. Build the wavetable side around band-limited mipmaps, adjacent-level crossfades, and high-quality interpolation so radical AI pitch or phase changes do not alias or click badly. Build the physical-modeling side around DaisySP-style primitives or equivalent structures, keep Karplus-Strong-style tuning on all-pass fractional delays when brightness matters, and place hard safety rails inside feedback paths: denormal protection, soft saturation, DC blocking, and fast NaN or Inf guards.

The modulation system should be treated as a core performance subsystem. Use a SoA-style matrix, SIMD-friendly fused loops, and background-built topology objects that swap into the live engine through atomic pointer exchange instead of in-place mutation. In JUCE, that points more toward `juce::dsp::SIMDRegister`-style fused loops than repeated helper passes across the same buffers.

Implement the Ear as a layered local pipeline. Start with continuous deterministic descriptors from Essentia, then add a constrained semantic tagger such as `sherpa-onnx` on larger windows if the Brain needs human-readable sound labels. Treat CLAP or TRR-style embedding branches as later additions for open-vocabulary or texture-sensitive reasoning rather than a first requirement. Aggregate Ear output into a compact structured payload and hand it to the Brain through a separate application or network queue.

For the Brain, prefer an out-of-process local model server such as Ollama by default. Use a streaming HTTP client suited to NDJSON responses, apply strict connection and read timeouts, and parse streamed chunks incrementally on a background thread. Treat embedded `llama.cpp` as the alternative when tighter packaging or offline self-containment matters more than memory isolation and crash containment; if embedded, statically link it, memory-map GGUF weights when possible, keep decode work on lowered-priority worker threads sized around physical CPU cores, and use grammar-constrained decoding for the final machine payload.

The Brain prompt should be bounded like a control system: explicit parameter ontology, valid ranges, few-shot examples, low-entropy decoding, and narrow grammars that disallow unknown keys. But structurally correct output is not enough. Sensitive DSP parameters such as cutoff, gain, pitch, or delay time still need appropriate smoothing or interpolation after the atomic read, otherwise the plugin will produce zipper noise or discontinuities.

For ONNX Runtime, dynamically link or prune the runtime, clamp intra-op and inter-op threading aggressively, and disable spinning so inference cannot starve the DAW. Build the synth engine from anti-aliased wavetable synthesis plus DaisySP-backed physical modeling, and reserve RTNeural for any tiny neural blocks that truly must run inline with statically allocated model state.

## Evidence

- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]] adds the strongest DSP-engine guidance so far for wavetable anti-aliasing, feedback safety, modulation layout, and topology swaps.
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]] adds the strongest control-path guidance so far for deterministic prompting, GBNF, APVTS routing, and smoothing.
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]] adds the most detailed Ear design so far, including layered analysis, local latency constraints, and staged queue handoff to the Brain.
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]] adds concrete build, packaging, and runtime guidance for `llama.cpp`, local HTTP streaming, ONNX Runtime, and RTNeural.
- [[wiki/sources/2026-05-01-audio-thread-safety-and-concurrency]] strengthens the thread model with guidance on memory ordering, false sharing, denormals, and APVTS routing.
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]] describes the overall architecture and the supporting libraries.
- [[wiki/concepts/modulation-matrix-architecture]] captures the routing and data-layout model for the DSP engine.
- [[wiki/concepts/ear-audio-analysis-pipeline]] captures the recommended layering of the Ear.
- [[wiki/concepts/realtime-safe-ai-audio-architecture]] captures the thread model and lock-free constraints.
- [[wiki/concepts/structured-parameter-mapping]] captures the JSON schema, grammar, and APVTS handoff pattern.
- [[wiki/concepts/hybrid-synthesis-engine]] captures the DSP backend strategy.
- [[wiki/entities/ollama]], [[wiki/entities/llama-cpp]], [[wiki/entities/essentia]], [[wiki/entities/onnx-runtime]], [[wiki/entities/sherpa-onnx]], [[wiki/entities/daisysp]], and [[wiki/entities/rtneural]] capture the tool tradeoffs.

## Implications

The safest implementation path is a staged build:

- first, build the synth engine shell and APVTS plumbing
- second, implement the alias-safe wavetable core
- third, add physical-model modules plus feedback safety rails
- fourth, add the modulation matrix and safe topology-swap infrastructure
- fifth, add Ear capture and bounded worker-thread infrastructure
- sixth, implement the Ear fast path with deterministic descriptors
- seventh, add direct DSP-state handoff and host-visible APVTS routing as separate pipelines
- eighth, add parameter smoothing and artifact-prevention paths for all sensitive DSP controls
- ninth, integrate local semantic tagging only where the fast path proves insufficient
- tenth, integrate an out-of-process Brain with explicit timeout and fallback behavior
- eleventh, evaluate advanced embeddings, embedded `llama.cpp`, and deeper ONNX deployment only once runtime budgets are measured

This keeps the project grounded in plugin stability before adding heavier AI behaviors.

## Related Pages

- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/modulation-matrix-architecture]]
- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]]
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-audio-thread-safety-and-concurrency]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]
