# Ingest Query Lint

## What It Is

These are the three recurring operations that govern the wiki's lifecycle:

- Ingest new sources into the maintained knowledge base.
- Query the existing wiki to answer questions.
- Lint the wiki for quality, consistency, and coverage.

## Why It Matters

The pattern only works if these operations are explicit and repeatable. They give the LLM a disciplined maintenance loop instead of ad hoc note taking.

## Key Details

- Ingest updates many pages, not just one summary.
- Query should start from `index.md` and prefer synthesized pages before raw sources.
- Lint keeps the wiki healthy by identifying contradictions, orphans, and missing pages.

## Related Pages

- [[wiki/concepts/llm-wiki]]
- [[wiki/concepts/persistent-synthesis]]
- [[wiki/analyses/first-ingest-example]]
- [[wiki/sources/2026-05-01-llm-wiki-idea-file]]

## Open Questions

- How often should lint passes be run for this vault?
