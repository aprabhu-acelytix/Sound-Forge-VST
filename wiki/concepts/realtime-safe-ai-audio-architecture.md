# Real-Time Safe AI Audio Architecture

## What It Is

This concept captures the thread model needed to combine AI inference with a real-time audio plugin: keep the audio callback bounded and lock-free, move analysis and inference to background workers, and apply parameter changes through the message thread.

## Why It Matters

The core risk in an AI-powered plugin is not model quality but real-time failure. Any blocking work, heap allocation, file I/O, mutex acquisition, or uncontrolled latency on the audio thread can cause dropouts and host instability.

## Key Details

- `processBlock` should handle DSP, lock-free buffer writes, and atomic or prepublished state reads only.
- Preallocate buffers in `prepareToPlay`; avoid dynamic allocation, locks, file or network I/O, and logging inside the audio callback.
- A worker thread or thread pool should run audio analysis, model inference, JSON parsing, and network calls.
- Use `juce::AbstractFifo` or equivalent SPSC structures for bounded streaming, and prefer acquire-release atomics for one-way signaling between worker and audio threads.
- Use `juce::ScopedNoDenormals` in the callback to prevent denormal-induced CPU spikes.
- The GUI/message thread should own APVTS writes, listener notifications, and `AsyncUpdater`, `Timer`, or `MessageManager::callAsync` callbacks originating from non-audio threads.
- Map model classes to distinct execution domains: heavyweight LLMs and ONNX models on background workers or external services, RTNeural only for very small inline models.
- Batch sizes for background analysis can be much larger than host callback sizes, but the batching and queueing policy must remain bounded.
- If analysis results need to reach the Brain over HTTP or another blocking transport, use a second queue and a separate application or network thread rather than stalling the analysis worker.
- Long-lived HTTP response streaming and NDJSON parsing belong entirely on background threads.
- Even correctly routed parameter updates can still be acoustically unsafe; the DSP path must smooth or interpolate sensitive controls after the atomic read.
- Larger DSP graph or modulation-matrix reconfigurations should be prepared off-thread and swapped atomically at a safe boundary rather than edited in place while the audio thread is running.
- The worker that builds a replacement topology should also own retirement of the old object after the swap so `processBlock` never performs teardown work.
- RealtimeSanitizer is a strong testing tool for catching hidden allocations or locks in supposedly real-time-safe code paths.

## Related Pages

- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/modulation-matrix-architecture]]
- [[wiki/entities/juce]]
- [[wiki/entities/onnx-runtime]]
- [[wiki/entities/rtneural]]
- [[wiki/entities/llama-cpp]]
- [[wiki/entities/ollama]]
- [[wiki/entities/essentia]]
- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]]
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]
- [[wiki/sources/2026-05-01-audio-thread-safety-and-concurrency]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- What CPU and latency budget should each background subsystem be allowed per plugin instance?
- How should the plugin degrade gracefully when a model server is unavailable or overloaded?
- Which state should bypass APVTS entirely and travel as direct DSP payloads instead?
- Which runtimes should be embedded inside the plugin versus isolated behind a local service boundary?
- Should Ear analysis and Brain transport live on separate workers from the start, or only once the pipeline grows more complex?
- Which message-thread update mechanism should be the default for AI-originated control changes: timer polling, async updates, or a hybrid?
- What is the safest swap boundary for larger engine-topology updates: sample, block, or explicit voice-cycle transitions?
