# Audio Thread Safety and Concurrency

## Summary

This source deepens the vault's concurrency model for JUCE plugins. It argues that real-time safety depends not just on moving AI off the audio thread, but on enforcing hard prohibitions against allocation, locks, logging, and I/O in `processBlock`, using bounded lock-free queues and atomics correctly, and routing APVTS updates through the message thread.

## Key Points

- The audio callback has a strict deadline and must avoid any unbounded operation.
- Dynamic allocation, mutexes, file or network I/O, and console logging are all unsafe in the callback.
- `juce::ScopedNoDenormals` is effectively mandatory protection against denormal-induced CPU spikes.
- `juce::AbstractFifo` is the preferred JUCE primitive for bounded SPSC transfer between threads.
- Acquire-release atomics are better suited than default sequential consistency for targeted cross-thread signaling.
- `juce::AsyncUpdater` is the safe bridge from background threads to APVTS and UI updates, but must never be triggered from the audio thread.
- RealtimeSanitizer is a useful testing tool for exposing hidden real-time violations.

## Claims

- Real-time audio safety is a concurrency problem as much as a DSP problem.
- Lock-free SPSC queues and POD payloads are the safest default for cross-thread data movement.
- APVTS interaction should be treated as a message-thread concern, not an audio-thread concern.
- False sharing, denormals, and overly strong atomic ordering can create performance failures even when code appears functionally correct.
- Background AI and UI work must be prevented from indirectly blocking the audio thread through priority inversion.

## Connections

- [[wiki/concepts/realtime-safe-ai-audio-architecture]] - Expands the real-time model with low-level concurrency and testing details.
- [[wiki/concepts/structured-parameter-mapping]] - Clarifies when to use APVTS versus direct DSP-state handoff.
- [[wiki/overview/ai-vst3-synthesizer-architecture]] - Strengthens the top-level architecture with more specific thread and queue guidance.
- [[wiki/entities/juce]] - Adds detail on `AbstractFifo`, `AsyncUpdater`, and `ScopedNoDenormals`.
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]] - Refines the current implementation recommendation.

## Open Questions

- Which parts of the plugin should use direct DSP-state queues instead of APVTS-backed parameters?
- Should RealtimeSanitizer become part of the project's default debug or CI workflow?
- Does the implementation need platform-specific audio-thread tuning for macOS workgroups or other host environments?

## Source

- Raw file: [[raw/sources/Audio Thread Safety and Concurrency]]
