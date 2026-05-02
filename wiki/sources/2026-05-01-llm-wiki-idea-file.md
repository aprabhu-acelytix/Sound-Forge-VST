# LLM Wiki Idea File

## Summary

This source proposes using an LLM as the maintainer of a persistent markdown wiki that accumulates synthesized knowledge over time. The wiki sits between raw sources and future answers, allowing cross-document structure, contradictions, and summaries to be maintained once and reused repeatedly.

## Key Points

- Standard RAG systems repeatedly rediscover knowledge from raw documents at query time.
- An LLM-maintained wiki preserves structure and synthesis across sessions.
- The architecture separates raw sources, generated wiki pages, and a schema document that constrains behavior.
- The recurring operations are ingest, query, and lint.
- `index.md` and `log.md` are key navigation artifacts for scale.
- Obsidian is a strong front end for browsing the maintained wiki.

## Claims

- A maintained wiki can compound in value as new sources and questions are integrated.
- The bookkeeping burden is the main reason humans abandon knowledge bases.
- LLMs are well-suited to cross-referencing and multi-file maintenance work.
- Durable answers should often be written back into the wiki as analysis pages.

## Connections

- [[wiki/concepts/llm-wiki]] - Defines the overall pattern described by the source.
- [[wiki/concepts/persistent-synthesis]] - Captures the central distinction from query-time retrieval.
- [[wiki/concepts/ingest-query-lint]] - Distills the source's repeated operational loop.
- [[wiki/entities/obsidian]] - Tracks the recommended browsing environment.
- [[wiki/overview/wiki-overview]] - Frames how this source seeds the current vault.

## Open Questions

- Which initial domains or projects should be ingested next?
- Should the vault add frontmatter conventions now or wait until there is a clearer querying need?
- When should local wiki search tooling be introduced?

## Source

- Raw file: [[raw/sources/2026-05-01-llm-wiki-idea-file]]
