# Essentia

## Overview

Essentia is a C++ music-information-retrieval and audio-analysis library with strong support for algorithmic timbral features such as MFCCs, spectral centroid, and transient descriptors.

## Relevance

The sources position Essentia as the default fast-path Ear because it can produce compact, deterministic feature summaries without needing heavyweight generative models. Its streaming mode makes it a strong fit for background analysis fed from a lock-free audio buffer.

Useful implementation details from the current sources:

- For plugin use, the streaming graph is the important mode, especially with `RingBufferInput` as the bridge from background worker code into the Essentia DAG.
- A practical Ear fast path is `RingBufferInput` -> `FrameCutter` -> `Windowing` -> `Spectrum`, with descriptor branches such as MFCC, spectral centroid, zero-crossing rate, and onset detection.
- Essentia is attractive because its descriptors are deterministic and relatively CPU-safe compared with deep models, making it a good always-on layer beneath slower semantic branches.
- Essentia is AGPL-licensed, so distribution and commercial-use implications may materially affect whether it can be adopted directly.

## Related Concepts

- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/ear-audio-analysis-pipeline]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-synth-ear-audio-analysis-pipeline]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- Which subset of features should form the stable timbre summary passed from the Ear to the Brain?
- Is Essentia's AGPL licensing compatible with the intended distribution model for this project?
