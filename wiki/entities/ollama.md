# Ollama

## Overview

Ollama is a local model server that exposes LLM inference through a REST API. In this vault it represents the default out-of-process deployment pattern for the Brain.

## Relevance

The sources recommend a local API server as the safest default for heavyweight language models because it isolates model memory, CPU load, and crashes from the DAW process. The plugin can talk to Ollama asynchronously through C++ HTTP code or a thin binding such as `ollama-hpp`.

Useful implementation details from the current sources:

- Ollama streams generation results as NDJSON over a long-lived HTTP connection.
- A background client should use strict connection and read timeouts so the plugin fails quickly if the service is unavailable.
- Incremental line-buffer parsing is needed because TCP chunk boundaries do not align with JSON object boundaries.
- `cpp-httplib` or `libcurl` are stronger fits than `juce::WebInputStream` for long-lived streamed responses.

## Related Concepts

- [[wiki/concepts/ear-brain-tutor-architecture]]
- [[wiki/concepts/realtime-safe-ai-audio-architecture]]
- [[wiki/concepts/structured-parameter-mapping]]

## Related Sources

- [[wiki/sources/2026-05-01-ai-inference-in-juce-plugins]]
- [[wiki/sources/2026-05-01-ai-powered-vst3-synthesizer-research]]

## Open Questions

- What retry, timeout, and offline fallback behavior should the plugin adopt when the local server does not respond?
- Should the project standardize on `cpp-httplib`, `libcurl`, or a different HTTP client for local model streaming?
