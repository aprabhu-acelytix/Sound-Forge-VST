# Log

## [2026-05-01] ingest | LLM Wiki Idea File

Summary:
- Created the initial wiki schema and vault structure.
- Ingested the user's LLM Wiki idea file as the seed source for the vault.
- Created initial overview, concept, entity, source, and analysis pages tied together through the index.

Files:
- [[AGENTS]]
- [[index]]
- [[log]]
- [[wiki/overview/wiki-overview]]
- [[wiki/concepts/llm-wiki]]
- [[wiki/concepts/persistent-synthesis]]
- [[wiki/concepts/ingest-query-lint]]
- [[wiki/entities/obsidian]]
- [[wiki/sources/2026-05-01-llm-wiki-idea-file]]
- [[wiki/analyses/first-ingest-example]]
- `raw/sources/2026-05-01-llm-wiki-idea-file.md`

Notes:
- This entry seeds the vault's initial operating model.
- Future sessions should follow the schema in `AGENTS.md` and extend it as needed.

## [2026-05-01] ingest | AI-Powered VST3 Synthesizer Research

Summary:
- Ingested a technical architecture source on combining JUCE DSP, local AI models, and lock-free thread boundaries inside a VST3 synthesizer.
- Added durable pages for the Ear-Brain-Tutor pattern, real-time-safe AI architecture, structured parameter mapping, and the hybrid synth engine.
- Added core library pages and a short implementation thesis that favors out-of-process LLM inference by default.

Files:
- [[AGENTS]]
- [[index]]
- [[log]]
- [[wiki/overview/wiki-overview]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/entities/juce]]
- [[wiki/entities/llama-cpp]]
- [[wiki/entities/ollama]]
- [[wiki/entities/onnx-runtime]]
- [[wiki/entities/essentia]]
- [[wiki/entities/daisysp]]
- [[wiki/entities/rtneural]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]
- `raw/sources/AI-Powered VST3 Synthesizer Research.md`

Notes:
- Current default thesis: keep all heavyweight inference off the audio thread and prefer an out-of-process model server unless embedded inference is justified.
- The next useful step is to turn this research into a concrete implementation plan or component map.

## [2026-05-01] ingest | Audio Thread Safety and Concurrency

Summary:
- Ingested a detailed concurrency source focused on hard real-time audio constraints, lock-free queueing, atomic memory ordering, and JUCE APVTS routing.
- Strengthened the existing architecture pages with explicit guidance around preallocation, `juce::ScopedNoDenormals`, `juce::AbstractFifo`, acquire-release atomics, and `juce::AsyncUpdater`.
- Clarified the difference between direct DSP-state handoff and host-visible APVTS updates.

Files:
- [[AGENTS]]
- [[index]]
- [[log]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/entities/juce]]
- [[wiki/sources/2026-05-01-audio-thread-safety-and-concurrency]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]
- `raw/sources/Audio Thread Safety and Concurrency.md`

Notes:
- Current vault position: use lock-free queues and atomics for bounded state transfer, and touch APVTS only from the Message Thread.
- Tooling and testing now matter more explicitly: hidden allocations, locks, and denormal slowdowns should be treated as architecture failures, not implementation details.

## [2026-05-01] ingest | AI Inference in JUCE Plugins

Summary:
- Ingested a source focused on concrete build, runtime, and deployment strategies for `llama.cpp`, local HTTP model servers, ONNX Runtime, and RTNeural inside JUCE plugins.
- Strengthened the existing runtime pages with integration details such as static versus dynamic linking, memory mapping, NDJSON streaming, ORT session constraints, and RTNeural initialization patterns.
- Clarified the current deployment recommendation: external model serving by default, embedded runtimes only with explicit resource and failure-isolation tradeoffs.

Files:
- [[AGENTS]]
- [[index]]
- [[log]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/entities/juce]]
- [[wiki/entities/llama-cpp]]
- [[wiki/entities/ollama]]
- [[wiki/entities/onnx-runtime]]
- [[wiki/entities/rtneural]]
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]
- `raw/sources/AI Inference in JUCE Plugins.md`

Notes:
- Current vault position: prefer external local API serving for heavyweight language models, constrain ONNX aggressively, and reserve RTNeural for tiny inline DSP networks.
- Build and packaging choices are now part of the architectural surface, not a separate implementation concern.

## [2026-05-01] ingest | AI Synth Ear Audio Analysis Pipeline

Summary:
- Ingested a detailed Ear-module source covering deterministic DSP descriptors, lightweight semantic tagging, texture-aware embeddings, and the staged queues that keep all of it off the audio thread.
- Added a dedicated Ear pipeline concept page and a `sherpa-onnx` entity page.
- Strengthened the existing architecture with clearer guidance on layered Ear design, local analysis latency, Essentia streaming, and ONNX deployment constraints.

Files:
- [[AGENTS]]
- [[index]]
- [[log]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/entities/essentia]]
- [[wiki/entities/onnx-runtime]]
- [[wiki/entities/sherpa-onnx]]
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]
- `raw/sources/AI Synth Ear_ Audio Analysis Pipeline.md`

Notes:
- Current vault position: the Ear should stay local, layered, and off-thread, with a deterministic fast path first and richer semantic branches added only as budgets allow.
- Essentia licensing and ONNX packaging are now explicit architectural selection criteria rather than afterthoughts.

## [2026-05-01] ingest | AI Synthesizer VST LLM GBNF JUCE

Summary:
- Ingested a control-plane source focused on deterministic prompt design, grammar-constrained JSON generation, safe JUCE/APVTS routing, and audio-rate parameter smoothing.
- Strengthened the parameter-mapping notes with explicit prompt, schema, and grammar guidance.
- Strengthened the runtime recommendation by making parameter smoothing and artifact prevention part of the Brain-to-DSP contract rather than a later DSP cleanup step.

Files:
- [[AGENTS]]
- [[index]]
- [[log]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/entities/juce]]
- [[wiki/entities/llama-cpp]]
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]
- `raw/sources/AI Synthesizer VST_ LLM, GBNF, JUCE.md`

Notes:
- Current vault position: deterministic JSON generation is necessary but not sufficient; the plugin must also smooth and apply those changes in a way that preserves audio quality.
- This source reinforces `juce::Timer`-based message-thread polling as a viable alternative to direct async callbacks for AI-originated parameter application.

## [2026-05-01] ingest | AI Synth DSP Architecture Deep Dive

Summary:
- Ingested a DSP-engine source focused on anti-aliased wavetable design, physical-model stability, custom sample-based graph choices, and high-throughput modulation routing.
- Added a dedicated modulation-matrix concept page.
- Strengthened the existing engine and runtime notes with clearer guidance on JUCE graph tradeoffs, DaisySP's sample-based strengths, wavetable mipmapping, feedback safety rails, and atomic topology swaps.

Files:
- [[AGENTS]]
- [[index]]
- [[log]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/modulation-matrix-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/entities/daisysp]]
- [[wiki/entities/juce]]
- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]
- `raw/sources/AI Synth DSP Architecture Deep Dive.md`

Notes:
- Current vault position: the core DSP engine should favor a custom sample-based graph around low-level primitives when the AI must rewire feedback-rich structures at runtime.
- Anti-aliasing, feedback safety, and modulation data layout are now treated as first-class architecture decisions rather than low-level implementation details.

## [2026-05-02] ingest | DSP Architecture Deep Dive Verification

Summary:
- Verified the `AI Synth DSP Architecture Deep Dive` ingest against the raw source after the interrupted session.
- Filled a few compressed details across the engine pages: mipmap crossfades, all-pass waveguide tuning, JUCE SIMD guidance, and worker-owned topology retirement after atomic swaps.

Files:
- [[log]]
- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/modulation-matrix-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/entities/daisysp]]
- [[wiki/entities/juce]]
- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]]

Notes:
- Open decisions are unchanged: first engine module scope, modulation density target, safest topology-swap boundary, and exact smoothing policies for sensitive DSP parameters.
