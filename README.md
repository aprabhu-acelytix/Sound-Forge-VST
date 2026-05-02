# Sound Forge VST

An Obsidian-based research vault for designing an AI-powered VST3 synthesizer.

## What This Repository Contains

- `AGENTS.md` defines the vault workflow and maintenance rules.
- `index.md` is the main navigation entry point.
- `log.md` records ingests, restructures, and verification work over time.
- `raw/` stores source material that is treated as immutable after ingestion.
- `wiki/` stores the maintained synthesis layer: concepts, entities, analyses, overviews, and source summaries.

## Current Focus

The current wiki centers on AI-powered VST3 synth architecture, including:

- JUCE plugin architecture
- real-time-safe AI integration
- Ear/Brain/Tutor system design
- local model runtime tradeoffs
- hybrid wavetable and physical-model DSP design
- modulation-matrix architecture and control-path safety

## Working In This Vault

New sessions should start by reading:

1. `AGENTS.md`
2. `index.md`
3. recent entries in `log.md`

The intended workflow is:

1. place source material in `raw/sources/`
2. ingest it into `wiki/`
3. update `index.md`
4. append `log.md`

## Notes

- `raw/` is the source-of-truth layer.
- `wiki/` is the maintained synthesis layer.
- Real-time safety, host stability, and DSP correctness are treated as first-order architectural constraints throughout the research.
