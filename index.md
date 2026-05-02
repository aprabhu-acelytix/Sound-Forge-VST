# Index

This is the content index for the vault. Read this first to orient to the current wiki.

## Overview

- [[wiki/overview/wiki-overview]] - High-level explanation of the vault, its architecture, and how to use it.
- [[wiki/overview/ai-vst3-synthesizer-architecture]] - Current architectural thesis for an AI-powered VST3 synthesizer built under real-time constraints.

## Concepts

- [[wiki/concepts/ear-brain-tutor-architecture]] - The agentic split between audio analysis, model-based control, and the user-facing conversational layer.
- [[wiki/concepts/ear-audio-analysis-pipeline]] - The layered local Ear path that turns synth audio into descriptors, semantic tags, and texture-aware context for the Brain.
- [[wiki/concepts/realtime-safe-ai-audio-architecture]] - The thread model and lock-free boundaries required to keep AI off the audio callback.
- [[wiki/concepts/structured-parameter-mapping]] - Constraining LLM outputs into deterministic parameter payloads that JUCE can apply safely.
- [[wiki/concepts/hybrid-synthesis-engine]] - Combining anti-aliased wavetable synthesis with physical modeling under AI control.
- [[wiki/concepts/modulation-matrix-architecture]] - Cache-friendly, SIMD-oriented modulation routing with safe topology swaps for an AI-driven synth engine.
- [[wiki/concepts/llm-wiki]] - The core pattern: an LLM incrementally maintains a persistent wiki from source material.
- [[wiki/concepts/persistent-synthesis]] - The wiki as a compiled synthesis layer rather than query-time rediscovery.
- [[wiki/concepts/ingest-query-lint]] - The three core operations that keep the knowledge base growing and healthy.

## Entities

- [[wiki/entities/juce]] - The core C++ framework for plugin structure, threading primitives, DSP utilities, and APVTS integration.
- [[wiki/entities/llama-cpp]] - Native local LLM inference engine for embedded background execution with grammar constraints.
- [[wiki/entities/ollama]] - Preferred out-of-process local model server for isolating heavy LLM inference from the DAW.
- [[wiki/entities/onnx-runtime]] - Background-only inference runtime for audio feature extraction and lightweight classifiers.
- [[wiki/entities/sherpa-onnx]] - C++ audio-tagging framework on top of ONNX Runtime, useful for lightweight Ear semantic labels.
- [[wiki/entities/essentia]] - Algorithmic MIR and feature-extraction library for the Ear fast path.
- [[wiki/entities/daisysp]] - Efficient DSP library that strengthens the physical-modeling side of the synth engine.
- [[wiki/entities/rtneural]] - Real-time-safe inline inference library for small neural DSP blocks.
- [[wiki/entities/obsidian]] - Obsidian as the browsing and editing environment for the maintained wiki.

## Analyses

- [[wiki/analyses/first-ingest-example]] - Worked example showing how the first source was ingested into the wiki.
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]] - Actionable synthesis of the current recommended stack and thread model.

## Sources

- [[wiki/sources/2026-05-01-ai-synth-dsp-architecture-deep-dive]] - Detailed DSP-engine guidance for wavetable anti-aliasing, physical-model safety, sample-based graph design, and SIMD-friendly modulation routing.
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]] - Detailed Brain/control-plane guidance for deterministic prompting, GBNF decoding, APVTS routing, and zipper-noise-safe parameter smoothing.
- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]] - Detailed Ear-module design covering Essentia streaming, lock-free capture, ONNX tagging, and texture-aware embeddings.
- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]] - Implementation-focused guidance for embedding or bridging local AI runtimes inside JUCE plugins, including `llama.cpp`, HTTP streaming, ONNX Runtime, and RTNeural.
- [[wiki/sources/2026-05-01-audio-thread-safety-and-concurrency]] - Detailed concurrency guidance for JUCE plugins covering the audio thread, lock-free queues, memory ordering, denormals, and APVTS routing.
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]] - Technical research on building an agentic VST3 synth with JUCE, local AI models, and hybrid DSP.
- [[wiki/sources/2026-05-01-llm-wiki-idea-file]] - Summary and extraction of the initial LLM Wiki idea file provided by the user.
