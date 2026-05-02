# AI Synthesizer VST LLM GBNF JUCE

## Summary

This source focuses on the Brain-to-DSP control path. It argues that reliable AI sound design requires a bounded prompt schema, grammar-constrained JSON generation, safe JUCE-side handoff into APVTS, and audio-rate smoothing after the control values reach the DSP engine.

## Key Points

- The Brain needs an explicit parameter ontology with ranges and examples, not just a high-level text request.
- Few-shot prompting and bounded reasoning improve mappings from sound-designer language to numeric parameter states.
- GBNF in `llama.cpp` can physically restrict JSON structure and allowed parameter keys at generation time.
- APVTS writes still belong on the message thread and should be fed from a lock-free queue.
- Message-thread timer polling is a viable alternative to async wakeups for draining AI-generated control updates.
- Even structurally valid parameter changes can create zipper noise or other artifacts unless the DSP path smooths them correctly.
- Multiplicative smoothing is often more appropriate than linear smoothing for logarithmic domains such as cutoff frequency or gain.

## Claims

- Deterministic prompting and grammar constraints are both required when the LLM is acting as a control system.
- Safe JUCE threading alone is insufficient; control data also needs acoustically safe application in the DSP loop.
- The control plane should be treated as a multi-stage contract: prompt bounds, generation bounds, thread-safe transport, and artifact-safe DSP reception.
- Low-entropy decoding is preferable when the output is executable control data rather than prose.

## Connections

- [[wiki/concepts/structured-parameter-mapping]] - Strengthens the existing control-data model with prompt and grammar details.
- [[wiki/concepts/realtime-safe-ai-audio-architecture]] - Adds more detail about message-thread draining and DSP-safe reception.
- [[wiki/entities/llama-cpp]] - Highlights GBNF as a major advantage of embedded local inference.
- [[wiki/entities/juce]] - Extends the JUCE-side application path with timer-based polling and smoothing concerns.
- [[wiki/overview/ai-vst3-synthesizer-architecture]] - Sharpens the top-level Brain-to-DSP control contract.
- [[wiki/analyses/recommended-architecture-for-ai-vst3-synth]] - Refines the staged implementation order.

## Open Questions

- Which parameter groups should have dedicated sub-grammars rather than one global object schema?
- Which JUCE update mechanism should be the default for AI-originated control changes: timer polling, async updates, or a hybrid?
- Which control domains need multiplicative smoothing first in the initial synth implementation?

## Source

- Raw file: [[raw/sources/AI Synthesizer VST_ LLM, GBNF, JUCE]]
