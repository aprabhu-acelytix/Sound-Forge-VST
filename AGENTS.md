# LLM Wiki Schema

This vault is a persistent, LLM-maintained wiki. The human curates sources, asks questions, and guides priorities. The LLM reads source material, updates the wiki, maintains cross-references, and keeps the knowledge base coherent over time.

## Mission

- Build and maintain a structured personal knowledge base as markdown files.
- Treat the wiki as a compiled synthesis layer between raw sources and answers.
- Preserve accumulation: do not re-derive everything from scratch when the wiki already contains vetted synthesis.
- Prefer updating existing pages over creating redundant new pages.

## Operating Model

There are three layers:

1. `raw/`
   Raw sources and attachments. These are immutable source-of-truth inputs. The LLM may read them but must not edit them after ingestion unless the human explicitly asks.
2. `wiki/`
   LLM-authored markdown pages. This is the maintained synthesis layer.
3. `AGENTS.md`
   This schema. It defines folder conventions, page formats, and required workflows.

## Vault Layout

```text
.
├── AGENTS.md
├── index.md
├── log.md
├── raw/
│   ├── assets/
│   └── sources/
└── wiki/
    ├── analyses/
    ├── concepts/
    ├── entities/
    ├── overview/
    ├── sources/
    └── templates/
```

## Folder Conventions

- `raw/sources/`
  Put source documents here. Examples: clipped articles, notes, transcripts, PDFs exported to markdown, book chapter notes.
- `raw/assets/`
  Local images and attachments referenced by source files.
- `wiki/overview/`
  High-level synthesis pages such as domain overviews, dashboards, roadmaps, and current theses.
- `wiki/concepts/`
  Topic pages for ideas, frameworks, methods, recurring themes, and abstractions.
- `wiki/entities/`
  Pages for people, organizations, products, books, projects, tools, places, or any durable named thing.
- `wiki/analyses/`
  Durable outputs created in response to questions: comparisons, research notes, timelines, decision memos, study guides.
- `wiki/sources/`
  One page per ingested source. These pages summarize the source, extract claims, and link into the rest of the wiki.
- `wiki/templates/`
  Optional helper templates and page skeletons maintained by the LLM.

## Root Files

- `index.md`
  Canonical content index for the whole wiki. Read this first when orienting. Keep it concise and scannable.
- `log.md`
  Append-only chronological log of ingests, queries, lint passes, and major restructures.

## Naming Rules

- Use readable markdown filenames in `kebab-case`.
- Prefer stable names that reflect the page subject, not the date.
- Prefix source summaries with a date when useful: `YYYY-MM-DD-short-title.md`.
- Keep page titles human-readable in Title Case.
- When in doubt, prefer updating an existing page with a strong exact match over making a near-duplicate.

## Page Types

### Source page

Stored in `wiki/sources/`.

Use this structure:

```md
# Source Title

## Summary

Short synthesis of the source.

## Key Points

- Important point

## Claims

- Claim or finding

## Connections

- Links to related wiki pages with one-line explanation

## Open Questions

- Unresolved issue or follow-up idea

## Source

- Raw file: [[path-or-note-name]]
```

### Concept page

Stored in `wiki/concepts/`.

Use this structure:

```md
# Concept Name

## What It Is

## Why It Matters

## Key Details

## Related Pages

## Open Questions
```

### Entity page

Stored in `wiki/entities/`.

Use this structure:

```md
# Entity Name

## Overview

## Relevance

## Related Concepts

## Related Sources

## Open Questions
```

### Analysis page

Stored in `wiki/analyses/`.

Use this structure:

```md
# Analysis Title

## Question

## Answer

## Evidence

## Implications

## Related Pages
```

## Link Policy

- Use Obsidian wiki links for all internal references.
- Add links while writing, not as a cleanup step.
- Prefer linking to durable concept or entity pages from source pages.
- When a page mentions an important concept repeatedly, create or update a dedicated page for it.
- Avoid orphan pages. Every new page should be linked from at least one existing page and from `index.md`.

## Source Handling Rules

- Treat files in `raw/` as immutable after placement.
- Preserve provenance. Every synthesized claim should be traceable to one or more source pages.
- If a new source contradicts existing wiki content, do not silently overwrite the old claim.
  Update affected pages to note the contradiction, name the competing claims, and link both supporting sources.
- Prefer quoting sparingly. Summarize in original wording unless exact phrasing matters.

## Standard Workflows

### 1. Ingest workflow

When the human asks to ingest a source:

1. Read the relevant source file from `raw/sources/` and any local assets if needed.
2. Identify whether the source is new or overlaps existing wiki pages.
3. Create or update the corresponding source page in `wiki/sources/`.
4. Update relevant concept, entity, overview, and analysis pages.
5. Update `index.md` to include any new pages or changed summaries.
6. Append an entry to `log.md` using the required format.
7. Report what changed, what remains uncertain, and which follow-up pages may be worth creating.

### 2. Query workflow

When answering a question:

1. Read `index.md` first to find candidate pages.
2. Read the relevant wiki pages.
3. Use raw sources only when the wiki is missing evidence or needs verification.
4. Answer with citations to wiki pages and sources.
5. If the answer is durable and likely to be useful later, file it into `wiki/analyses/`, update `index.md`, and append to `log.md`.

### 3. Lint workflow

When asked to lint or health-check the wiki:

1. Scan `index.md`, `log.md`, and representative wiki pages.
2. Look for contradictions, stale claims, duplicate pages, orphan pages, weak summaries, broken links, and obvious gaps.
3. Fix straightforward issues directly.
4. Record the lint pass in `log.md`.
5. Return findings prioritized by impact.

## Log Format

Each entry in `log.md` must start with this heading format:

```md
## [YYYY-MM-DD] operation | Title
```

Where `operation` is one of:

- `ingest`
- `query`
- `lint`
- `restructure`

Each log entry should contain:

- `Summary`: one short paragraph or 1-3 bullets
- `Files`: affected files as wiki links or paths
- `Notes`: optional unresolved issues or next steps

## Index Format

`index.md` should stay compact. Organize it by category. Each entry should have:

- a wiki link
- a one-line description

Suggested sections:

- Overview
- Concepts
- Entities
- Analyses
- Sources

## Maintenance Rules

- Prefer minimal edits that keep the wiki coherent.
- Do not create placeholder pages without at least a useful seed summary.
- If a page becomes too broad, split it into two or more focused pages and update links.
- Preserve chronology in `log.md`; never rewrite past entries except to fix formatting mistakes.
- Periodically improve this `AGENTS.md` when patterns emerge or the human requests a workflow change.

## Collaboration Rules

- Ask for clarification only when a decision materially affects structure or meaning.
- Otherwise, make the smallest reasonable choice and keep moving.
- Explain significant wiki updates briefly after making them.
- Assume future sessions should continue following this schema unless the human changes it.

## First-Step Default

At the start of a new session in this vault:

1. Read `AGENTS.md`.
2. Read `index.md`.
3. Read recent entries in `log.md`.
4. Then perform the requested operation.

## Current Domain

This vault now focuses on AI-powered VST3 synthesizer architecture and implementation research, especially JUCE plugin design, real-time-safe AI integration, DSP engine design, local model runtimes, and agentic sound-design workflows. The earlier LLM Wiki pages remain part of the vault's operating infrastructure and maintenance pattern.

## Domain-Specific Notes

- Treat real-time safety and host stability as first-order constraints when summarizing or evaluating technical choices.
- Prefer durable pages around thread boundaries, parameter-state handoff, inference placement, DSP building blocks, and library tradeoffs.
- When a source presents both a recommended default and a viable alternative, capture both explicitly.
- Distinguish between internal DSP-state handoff and host-visible parameter automation: POD state may move across lock-free queues or atomics, while APVTS writes and listener notifications belong on the Message Thread.
- For each AI runtime or deployment option, capture four things explicitly when the source supports it: build and linking method, execution domain, memory or loading strategy, and failure-isolation boundary.
- Capture licensing or distribution constraints when they materially affect library selection, plugin packaging, or commercial viability.
- For AI control paths, capture both control-plane correctness and DSP-plane behavior: a payload can be structurally valid yet still produce audible artifacts if smoothing, coefficient updates, or automation semantics are wrong.
- For DSP-engine sources, capture four things explicitly when the source supports it: anti-aliasing strategy, feedback or stability safeguards, graph or routing flexibility, and modulation data layout or vectorization strategy.
