# RTNeural

## Overview

RTNeural is a C++ inference library designed for real-time-safe execution of small neural networks after initialization.

## Relevance

The sources treat RTNeural as the exception to the general rule that neural inference must stay off the audio thread. It is appropriate only for micro-neural DSP blocks that must live directly in the signal path, not for large language models or heavyweight audio classifiers.

Useful implementation details from the current sources:

- The strongest fit is compile-time-defined network shapes with statically allocated memory.
- Model weights should be loaded during initialization, not inside `processBlock`.
- Recurrent or stateful models should be reset in `prepareToPlay` or equivalent transport lifecycle hooks.
- Backend choice such as STL, XSIMD, or Eigen should follow model size and latency constraints.

## Related Concepts

- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/hybrid-synthesis-engine]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Is there a concrete inline neural DSP block worth prioritizing in the first implementation pass?
- Which RTNeural backend is the best fit for the first intended inline model?
