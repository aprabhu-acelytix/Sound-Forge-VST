# First Ingest Example

## Question

What does the first complete ingest look like for this vault?

## Answer

The first ingest used the user's LLM Wiki idea file as a seed source. The raw text was preserved in `raw/sources/`, summarized in `wiki/sources/`, split into durable concepts in `wiki/concepts/`, linked to an entity page for Obsidian, surfaced in an overview page, indexed in `index.md`, and recorded chronologically in `log.md`.

## Evidence

- [[raw/sources/2026-05-01-llm-wiki-idea-file]] stores the immutable source text.
- [[wiki/sources/2026-05-01-llm-wiki-idea-file]] captures the source summary, claims, and connections.
- [[wiki/concepts/llm-wiki]], [[wiki/concepts/persistent-synthesis]], and [[wiki/concepts/ingest-query-lint]] extract reusable ideas from the source.
- [[wiki/entities/obsidian]] captures the main tool named in the source.
- [[index]] and [[log]] make the new knowledge navigable and traceable.

## Implications

This demonstrates the intended maintenance pattern: one source can update many pages, and the resulting synthesis becomes the starting point for future questions and future ingests.

## Related Pages

- [[wiki/sources/2026-05-01-llm-wiki-idea-file]]
- [[wiki/overview/wiki-overview]]
- [[wiki/concepts/ingest-query-lint]]
