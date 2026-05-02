# JUCE

## Overview

JUCE is the central C++ framework for building the plugin. In this vault it anchors the processor/editor split, `processBlock`, threading primitives, APVTS state management, and much of the baseline DSP infrastructure.

## Relevance

The research sources assume JUCE as the host framework for nearly every critical boundary: `processBlock` execution, lock-free FIFOs through `juce::AbstractFifo`, APVTS automation, `AsyncUpdater` dispatch onto the message thread, `MessageManager::callAsync` for lightweight UI callbacks from background workers, `Timer`-based polling on the message thread, `ScopedNoDenormals` protection in the callback, and selective use of the `juce_dsp` module for filters, SIMD utilities, and general DSP support. The sources also show that JUCE-based CMake projects may need extra care when embedding external runtimes such as `llama.cpp`, especially around source grouping and third-party submodule integration.

Current nuance from the DSP-engine sources:

- `juce::dsp::ProcessorChain` is strong for fixed compile-time signal paths but is too rigid for AI-directed graph rewiring.
- `juce::AudioProcessorGraph` permits runtime graph changes but is a poor fit for sample-level feedback-heavy synthesis architectures.
- `juce::AudioProcessorGraph` also protects itself by zeroing cyclic feedback paths, which conflicts with deliberate waveguide and resonator loops.
- `juce::dsp::SIMDRegister` is the stronger JUCE abstraction when the modulation matrix needs fused SIMD math in one pass; repeated `juce::FloatVectorOperations` passes can become memory-bound.
- JUCE is therefore strongest as the plugin shell and utility layer around a custom sample-based engine when the synth graph must be highly dynamic.

## Related Concepts

- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]
- [[wiki/concepts/hybrid-synthesis-engine]]
- [[wiki/concepts/modulation-matrix-architecture]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]]
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]]
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-audio-thread-safety-and-concurrency]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Which JUCE primitives should become standard across the implementation: `AbstractFifo`, `AsyncUpdater`, `MessageManager::callAsync`, `ThreadPool`, `AudioWorkgroup`, or custom wrappers?
- Should AI-originated parameter application standardize on `Timer` polling for message-thread draining rather than async wakeups?
- How much of the core synth engine should rely on JUCE DSP blocks versus a custom sample-based graph around lower-level primitives?
