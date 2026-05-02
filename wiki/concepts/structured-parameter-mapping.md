# Structured Parameter Mapping

## What It Is

Structured parameter mapping is the process of turning user intent and audio context into a deterministic parameter payload that the synthesizer can apply safely.

## Why It Matters

Free-form LLM text is fragile in software control loops. The source argues that a usable AI sound-design system must emit machine-readable data with explicit bounds, enums, and stable formatting.

## Key Details

- Prompts should define the model's role, the available parameters, and the valid ranges.
- Few-shot examples improve mappings from sonic language to parameter values.
- Controlled reasoning or explicit analysis blocks can improve acoustic mapping, but the final machine-readable payload should stay compact and bounded.
- Grammar-constrained decoding, such as GBNF in `llama.cpp`, prevents malformed output and can physically disallow unknown parameter keys.
- Low temperature and restricted sampling help reduce entropy when the model is acting as a control system rather than a creative co-writer.
- Host-visible parameter payloads should be queued from a worker thread and applied on the message thread using APVTS change gestures.
- Internal DSP-only state can bypass APVTS and travel as POD payloads or acquire-release-published structs when host automation is not required.
- `triggerAsyncUpdate()` is appropriate from a background thread but must never be called from the audio thread.
- The audio thread should consume host-visible values through `getRawParameterValue()` atomics and other prepublished state through lock-free reads only.
- A valid payload still needs DSP-safe application: abrupt parameter jumps can create zipper noise unless the receiving parameter path is smoothed appropriately.

## Related Pages

- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/entities/juce]]
- [[wiki/entities/llama-cpp]]
- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/sources/2026-05-01-ai-synthesizer-vst-llm-gbnf-juce]]
- [[wiki/sources/2026-05-01-audio-thread-safety-and-concurrency]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Which parameter schema should define the first production-safe sound-design surface?
- When should explanations or reasoning traces be preserved versus discarded before control data is applied?
- Which controls need DAW-visible automation semantics, and which should remain internal macro state?
- Which parameters need multiplicative smoothing, linear smoothing, or custom interpolation paths in the DSP engine?
