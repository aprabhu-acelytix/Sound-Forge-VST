# AI-Powered VST3 Synthesizer Architecture

## Thesis

An agentic VST3 synthesizer is viable only if the system treats real-time safety as non-negotiable. The audio thread must stay pure DSP and lock-free state reads, while AI inference, feature extraction, and network communication execute off-thread and hand results back through bounded queues and message-thread parameter application.

## Recommended Stack

- JUCE for plugin structure, `processBlock`, APVTS, `AbstractFifo`, `AsyncUpdater`, and core DSP utilities.
- An Ear-Brain-Tutor split where the Ear analyzes audio, the Brain maps language and sonic context to parameters, and the Tutor presents the conversational interface.
- Essentia as the default fast-path Ear for deterministic timbral features.
- A layered local Ear pipeline: continuous Essentia descriptors, slower ONNX-based tagging on larger windows, and optional CLAP or TRR-style embeddings only when richer semantic or textural context is needed.
- Keep Ear-to-Brain handoff structured and asynchronous: format a compact payload on the analysis side and pass it through a separate application or network queue rather than blocking the Ear worker on HTTP.
- Ollama or another local API server as the default Brain deployment model, with embedded `llama.cpp` as the main alternative.
- For embedded `llama.cpp`, prefer static linking, memory-mapped GGUF loading, and lowered-priority worker threads sized around physical CPU cores.
- For local API serving, prefer streaming HTTP clients such as `cpp-httplib` or `libcurl` over `juce::WebInputStream` for long-lived NDJSON token streams.
- For ONNX Runtime, dynamically link prebuilt or pruned binaries, clamp session threading aggressively, and disable thread spinning.
- Deterministic Brain control should use tightly bounded prompts, few-shot examples, low-entropy decoding, and grammar-constrained JSON generation.
- Brain-to-DSP application should include message-thread-safe APVTS routing plus audio-rate smoothing for artifact-free transitions.
- Use JUCE for the plugin shell and selected DSP utilities, but favor a custom sample-based engine graph when the AI must rewire or feed back synthesis blocks at runtime.
- The wavetable core should use FFT-based mipmapping, band-limited table selection, and high-quality interpolation rather than naive lookup.
- The physical-modeling side should include explicit safety rails such as denormal mitigation, soft saturation, DC blocking, and NaN or Inf guards.
- The modulation system should use a SoA-style, SIMD-friendly matrix and swap larger topology changes through atomic pointer exchange rather than in-place mutation.
- Preallocated buffers, lock-free FIFOs, and acquire-release atomics for cross-thread handoff.
- `juce::ScopedNoDenormals` in the audio callback and RealtimeSanitizer in testing to catch hidden real-time violations.
- A hybrid synth engine built from anti-aliased wavetable synthesis plus DaisySP-style physical modeling.
- RTNeural only where a very small neural block must run directly on the audio thread.

## Main Risks

- Audio dropouts caused by blocking work, heap allocation, or thread contention in `processBlock`.
- Audio instability caused by hidden denormal slowdowns, false-sharing-heavy queues, or overuse of sequentially consistent atomics.
- Host instability if APVTS writes occur from the wrong thread.
- Excess CPU usage from ONNX Runtime thread spinning, oversized local models, or poorly sized `llama.cpp` worker counts.
- Binary bloat, memory fragmentation, or direct DAW crash coupling from in-process model deployment.
- Aliasing from naive wavetable lookup or poorly band-limited pitch modulation.
- AVX throttling and cache eviction from heavyweight inference workloads competing with DSP.
- Stale Brain decisions if Ear analysis becomes too heavyweight or remote enough that it no longer reflects the current sound.
- Licensing or deployment constraints from AGPL analysis libraries and dynamically packaged ONNX dependencies.
- Structurally valid but acoustically unsafe parameter jumps that create zipper noise, discontinuities, or expensive coefficient churn.
- Graph-mutation races or unstable feedback paths when the AI rewires synthesis topology without safe swap semantics.
- NaN, Inf, DC buildup, or runaway feedback energy in physical-model loops.
- Unstable or speaker-dangerous behavior in poorly bounded physical-model parameter spaces.

## Related Pages

- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/modulation-matrix-architecture]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]
- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]]
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]
