# Wiki Overview

## What This Vault Is

This vault is a persistent, LLM-maintained wiki built from raw source material. Instead of repeatedly searching the same documents at question time, the wiki stores accumulated synthesis in markdown pages that can be revised as new evidence arrives.

## Architecture

- `raw/` stores immutable source material.
- `wiki/` stores generated synthesis pages.
- `AGENTS.md` defines the maintenance rules and workflows.
- `index.md` is the content-oriented navigation layer.
- `log.md` is the chronological record of operations.

## How To Use It

- Add new source files to `raw/sources/`.
- Ask the LLM to ingest them one at a time or in batches.
- Ask questions against the wiki.
- File durable answers back into `wiki/analyses/`.
- Periodically lint the vault for contradictions, gaps, and stale pages.

## Current State

The vault now has two layers of content:

- infrastructure pages that define how the LLM-maintained wiki operates
- project research pages focused on building an AI-powered VST3 synthesizer

The current technical thesis is that the synthesizer should isolate AI work from the audio callback, use lock-free and message-thread handoff patterns for control data, and combine a hybrid DSP engine with constrained model outputs.

## Related Pages

- [[wiki/overview/ai-vst3-synthesizer-architecture]]
- [[wiki/concepts/llm-wiki]]
- [[wiki/concepts/persistent-synthesis]]
- [[wiki/concepts/ingest-query-lint]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/entities/obsidian]]
- [[wiki/sources/2026-05-01-llm-wiki-idea-file]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]
