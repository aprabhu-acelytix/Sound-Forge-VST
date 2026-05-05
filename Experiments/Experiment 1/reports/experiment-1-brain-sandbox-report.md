# Experiment 1: The Brain Sandbox Report

## Purpose

This report summarizes the design, execution, and findings of `Experiment 1: The Brain Sandbox` for the Sound Forge Brain module prototype.

The goal of this experiment was to test whether a local LLM can act as a deterministic sound-design control surface rather than a conversational assistant. Specifically, the experiment evaluated whether a local model can convert natural-language sound design requests into a strict JSON payload for a bounded synthesizer parameter surface, while avoiding malformed output and avoiding conversational spillover.

This work is an early control-plane validation step for a future JUCE/C++ implementation of the Sound Forge Brain.

## Research Question

Can a local LLM, running under grammar-constrained decoding, reliably generate 100% syntactically valid JSON for a bounded synthesizer parameter schema without hallucinating extra conversational text?

Secondary questions:

- Can this be done locally on consumer hardware with CUDA GPU offload?
- What latency envelope is observed for small and medium instruct models?
- Does a stronger local model materially improve robustness under adversarial prompting?

## Parameter Surface Under Test

The experiment used a deliberately small synth parameter surface with exactly five required keys:

- `osc_shape`: one of `sine`, `saw`, `square`
- `filter_cutoff`: float in `20.0..20000.0`
- `attack_ms`: float in `0.1..5000.0`
- `decay_ms`: float in `1.0..10000.0`
- `reverb_mix`: float in `0.0..1.0`

The output contract required:

- one JSON object only
- exact key order
- no extra keys
- no markdown
- no code fences
- no explanations
- no text before or after the JSON object

## Experimental Architecture

The experiment was implemented in Python using `llama-cpp-python` with GGUF models and a GBNF grammar.

### Why this stack was selected

- `llama.cpp` provides grammar-constrained decoding, which is central to the hypothesis.
- GGUF supports quantized local models suitable for consumer hardware.
- The stack is aligned with the likely long-term local deployment direction for a Sound Forge Brain prototype.
- CUDA offload through `llama.cpp` makes it possible to test practical local performance on the project machine.

### Constraint strategy

Two layers of enforcement were used:

1. Generation-time constraint
- A GBNF grammar restricted the model to the exact JSON structure, exact key names, and exact oscillator enum values.

2. Post-parse validation
- Python validation enforced:
  - required keys
  - exact key order
  - numeric finiteness
  - numeric range limits

This separation is important. The grammar solved structural correctness. The validator solved semantic boundary enforcement.

## Prompting Strategy

The system prompt instructed the model to act as an expert synth sound designer and parameter-mapping engine. It explicitly defined:

- the parameter ontology
- valid ranges
- required key order
- a short set of sound-design heuristics
- silent internal reasoning only
- final output must be the JSON object only

The system prompt did not rely on the model to voluntarily behave. It worked together with the GBNF grammar to reduce output entropy and physically prevent most formatting failures.

## Runtime Environment

### Host execution path

The experiment was ultimately run outside the Docker sandbox because the sandbox runtime did not support GPU passthrough. Instead, the experiment used a standard host Docker GPU workflow with:

- `nvidia/cuda:12.6.3-devel-ubuntu22.04`
- CUDA-enabled `llama-cpp-python`
- host bind-mount of the repo
- host bind-mount of the GGUF model directory

### GPU confirmation

The successful runs explicitly reported CUDA GPU discovery:

- `ggml_cuda_init: found 1 CUDA devices`
- detected GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- compute capability: `8.9`
- VRAM: about `8187 MiB`

This confirms that the completed experiment runs used CUDA GPU offload rather than CPU-only inference.

## Models Evaluated

The following GGUF models were tested:

1. `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
2. `Qwen2.5-7B-Instruct-Q4_K_M.gguf`

These were chosen because:

- Qwen 2.5 is strong at instruction following and structured outputs.
- 3B provides a fast plumbing validation model.
- 7B provides a stronger comparison point for semantic robustness.

## Fixture Sets

Two prompt sets were used.

### 1. Baseline fixture set

File:

- `Experiments/Experiment 1/fixtures/test_prompts.json`

This set contained ten normal sound-design prompts such as:

- acid bass
- ambient pad
- glass pluck
- soft sub
- retro lead
- cinematic swell
- dry stab
- mellow keys
- washed drone
- mixed-intent hybrid prompt

### 2. Adversarial fixture set

File:

- `Experiments/Experiment 1/fixtures/adversarial_prompts.json`

This set contained ten robustness-oriented prompts designed to stress the control path:

- conflicting acoustic instructions
- prompt injection attempts
- requests for markdown/YAML
- requests for extra keys
- explicit out-of-range numeric requests
- unsupported enum pressure such as asking for a triangle wave
- narrative reasoning pressure
- under-specified prompts
- multilingual phrasing
- quote-heavy strings

## Measurement Method

For every prompt, the experiment recorded:

- raw output
- parsed payload
- syntax validity
- schema validity
- range validity
- key-order validity
- finish reason
- prompt/completion token counts
- tokens per second
- per-prompt latency in milliseconds

The experiment also recorded:

- model load time
- mean latency
- p95 latency
- p99 latency
- failure-category counts

Inference latency was measured with `time.perf_counter()`.

## Relevant Run Artifacts

### Smoke test

- `Experiments/Experiment 1/results/run_20260504T014841Z.json`

### Baseline full runs

- `Experiments/Experiment 1/results/run_20260504T034027Z.json` - Qwen 2.5 3B baseline
- `Experiments/Experiment 1/results/run_20260504T034057Z.json` - Qwen 2.5 7B baseline

### Adversarial full runs

- `Experiments/Experiment 1/results/run_20260504T034154Z.json` - Qwen 2.5 3B adversarial
- `Experiments/Experiment 1/results/run_20260504T034228Z.json` - Qwen 2.5 7B adversarial

### Status logs

- `Experiments/Experiment 1/logs/status_qwen3b_baseline.log`
- `Experiments/Experiment 1/logs/status_qwen7b_baseline.log`
- `Experiments/Experiment 1/logs/status_qwen3b_adversarial.log`
- `Experiments/Experiment 1/logs/status_qwen7b_adversarial.log`

## Results

## Baseline Results

### Qwen 2.5 3B Q4_K_M baseline

Artifact:

- `run_20260504T034027Z.json`

Summary:

- total prompts: `10`
- overall pass: `10/10`
- syntax pass: `10/10`
- schema pass: `10/10`
- range pass: `10/10`
- key-order pass: `10/10`

Latency:

- model load: `12911.78 ms`
- mean prompt latency: `1494.22 ms`
- median prompt latency: `1497.05 ms`
- p95 prompt latency: `1600.54 ms`
- min/max prompt latency: `1354.40 ms` / `1618.87 ms`

Interpretation:

- The 3B model was fully stable across the baseline fixture set.
- For normal sound-design requests, it produced strictly valid machine payloads with no observed structure failures.

### Qwen 2.5 7B Q4_K_M baseline

Artifact:

- `run_20260504T034057Z.json`

Summary:

- total prompts: `10`
- overall pass: `10/10`
- syntax pass: `10/10`
- schema pass: `10/10`
- range pass: `10/10`
- key-order pass: `10/10`

Latency:

- model load: `32618.12 ms`
- mean prompt latency: `2123.34 ms`
- median prompt latency: `2088.72 ms`
- p95 prompt latency: `2305.32 ms`
- min/max prompt latency: `1959.91 ms` / `2322.03 ms`

Interpretation:

- The 7B model was also fully stable across the baseline fixture set.
- It incurred higher load time and prompt latency than the 3B model, but maintained perfect structural and semantic-boundary compliance on the baseline set.

## Adversarial Results

### Qwen 2.5 3B Q4_K_M adversarial

Artifact:

- `run_20260504T034154Z.json`

Summary:

- total prompts: `10`
- overall pass: `9/10`
- syntax pass: `10/10`
- schema pass: `10/10`
- range pass: `9/10`
- key-order pass: `10/10`

Latency:

- model load: `15531.43 ms`
- mean prompt latency: `1576.81 ms`
- median prompt latency: `1434.70 ms`
- p95 prompt latency: `2147.94 ms`
- max prompt latency: `2469.77 ms`

Failure categories:

- `pass`: `9`
- `range_fail`: `1`

Observed failure:

- `out_of_range_bait_001`
  - the prompt explicitly pressured the model to output illegal values
  - returned `filter_cutoff = 50000.0`
  - this violated the allowed maximum of `20000.0`
  - the validator correctly rejected it

Important nuance:

- The model still remained structurally disciplined under adversarial pressure.
- The failure was not malformed JSON.
- The failure was semantic boundary obedience under hostile numeric prompting.

### Qwen 2.5 7B Q4_K_M adversarial

Artifact:

- `run_20260504T034228Z.json`

Summary:

- total prompts: `10`
- overall pass: `10/10`
- syntax pass: `10/10`
- schema pass: `10/10`
- range pass: `10/10`
- key-order pass: `10/10`

Latency:

- model load: `38174.67 ms`
- mean prompt latency: `2198.58 ms`
- median prompt latency: `2061.27 ms`
- p95 prompt latency: `2802.89 ms`
- max prompt latency: `3138.35 ms`

Interpretation:

- The 7B model remained compliant even under the adversarial numeric-pressure case.
- In the out-of-range bait prompt it produced clamped boundary values instead of illegal values:
  - `filter_cutoff = 20000.0`
  - `attack_ms = 0.1`
  - `decay_ms = 10000.0`
  - `reverb_mix = 1.0`

This is a meaningful robustness improvement over the 3B model.

## Adversarial Prompt Behavior Observations

Several results are worth highlighting because they reveal the shape of the control-system behavior.

### Prompt injection and formatting attacks

Prompts asked for:

- markdown code blocks
- YAML output
- extra explanation
- step-by-step reasoning

Observed behavior:

- both models still returned a parseable JSON object only
- no extra explanatory prose survived the grammar constraints
- some outputs ended with trailing whitespace and reached `finish_reason = length`
- the trimmed outputs remained valid and passed

Interpretation:

- The grammar was effective at preventing format drift.
- The `finish_reason = length` cases suggest the generation budget can still be tightened for efficiency.

### Extra-key pressure

Prompts requested unsupported keys such as:

- resonance
- sustain
- chorus depth
- tempo sync

Observed behavior:

- both models emitted only the allowed five keys

Interpretation:

- The grammar successfully disallowed ontology expansion.

### Unsupported enum pressure

Prompt requested:

- triangle wave

Observed behavior:

- the models selected one of the permitted oscillator values instead

Interpretation:

- The enum restriction was effective.

### Explicit illegal numeric pressure

Prompt requested:

- cutoff `50000 Hz`
- attack `0 ms`
- decay `200000 ms`
- reverb `150 percent`

Observed behavior:

- 3B let one illegal value through
- 7B mapped the request back into the legal range boundaries

Interpretation:

- Model quality still matters after grammar is solved.
- Post-parse validation remains mandatory.

## Comparative Interpretation

## What has been convincingly demonstrated

This experiment now supports the following conclusions:

1. A local LLM can be used as a structured control generator rather than a conversational assistant.
2. Grammar-constrained decoding can effectively eliminate malformed JSON and suppress extra conversational text for this parameter surface.
3. Consumer hardware with CUDA GPU offload can run this workflow locally with practical latency for non-real-time control generation.
4. A stronger local model materially improves robustness under adversarial prompting, even when structural correctness is already solved.
5. Grammar alone is not enough; semantic boundary checks after parsing are still required.

## What has not been fully demonstrated

The experiment does not yet prove:

1. that the generated parameter values are acoustically optimal
2. that the mapping quality is sufficient across a much larger and more diverse prompt corpus
3. that this latency is acceptable for all target user interactions in the eventual product
4. that the same behavior will hold identically when embedded inside a future C++ `llama.cpp` integration rather than a Python host container
5. that this parameter surface is rich enough for a useful first production Brain

## Why the results still matter

Even with those limitations, this experiment clears a major technical uncertainty.

The original question was whether the Brain concept could be made deterministic enough to emit machine-consumable parameter payloads instead of drifting into chatbot behavior. On that question, the answer is now strongly positive.

The experiment moved the project from architectural speculation to a working local prototype with measured evidence.

## Limitations

The following limitations should be noted for advisor review:

1. Small parameter surface
- only five parameters were tested
- this is sufficient for a control-path proof but not for a full synth design surface

2. Limited semantic evaluation
- outputs were validated structurally and numerically
- no formal human scoring rubric was applied for acoustic plausibility

3. Limited prompt count
- baseline set: ten prompts
- adversarial set: ten prompts
- useful, but still relatively small for a strong generalization claim

4. Post-parse trimming behavior
- some adversarial runs hit `finish_reason = length` and produced trailing whitespace
- these did not break the parser after trimming, but they are worth tightening in the next iteration

5. Python host implementation
- the prototype demonstrates viability, but final deployment constraints for JUCE/C++ will differ

## Practical Implications for Sound Forge

These findings suggest the following project implications.

1. The Brain should not emit free-form text into the control plane.
- Strict grammar-constrained structured output is the correct pattern.

2. The plugin should always validate the payload after generation.
- Structural validity is necessary but insufficient.

3. If the project wants stronger robustness under adversarial or ambiguous prompting, 7B-class local models are preferable to 3B-class models.

4. The current measured latencies are appropriate only for background-thread control generation, not the audio thread.
- This is consistent with the project architecture thesis.

5. The system should treat grammar, validation, and downstream DSP-safe application as separate contracts.

## Recommended Next Experiments

The next experiment should not simply repeat this one. It should target the remaining unknowns.

### 1. Semantic evaluation pass

Add a scoring rubric for whether the parameter choices are acoustically plausible.

Suggested dimensions:

- oscillator appropriateness
- brightness plausibility
- envelope plausibility
- reverb plausibility
- overall mapping coherence

### 2. Clamp-versus-reject policy experiment

The current validator rejects out-of-range values.

Future test:

- compare strict reject behavior against safe post-parse clamping
- assess whether clamping improves product robustness without masking systematic model weakness

### 3. Larger and more systematic prompt corpus

Expand the fixture library to include:

- more ambiguous prompts
- more adversarial numeric requests
- more multilingual prompts
- more unsupported feature requests
- repeated prompts with paraphrase variations

### 4. Stronger model/quant comparison

Evaluate whether a slightly stronger quant or model tier materially improves semantic quality enough to justify the latency cost.

Candidate next comparison:

- `Qwen2.5-7B-Instruct-Q5_K_M.gguf`

### 5. C++/embedded parity experiment

If the long-term deployment path points toward embedded `llama.cpp`, repeat the same core benchmark from a thin C++ host to confirm behavior and latency outside Python.

## Final Conclusion

Experiment 1 successfully demonstrated that the Sound Forge Brain concept is viable as a grammar-constrained local control system.

The strongest validated conclusion is:

> A local LLM can be made to emit deterministic, syntactically strict, schema-compliant synthesizer parameter JSON with no extra conversational text, and this can be done locally on consumer NVIDIA hardware with measured latency and reproducible artifacts.

The experiment also showed that model size still matters after structural constraints are imposed. The 7B model was not necessary to achieve structural correctness, but it was meaningfully better at resisting adversarial numeric pressure.

Accordingly, the project should treat the following as the current best design position:

- use grammar-constrained decoding
- always validate after parse
- keep inference off the audio thread
- prefer the stronger 7B model when robustness matters more than latency

For advisor review, the key open question is no longer whether this structured-output Brain approach works at all. The remaining questions are about semantic quality, latency budgets, deployment path, and product-level robustness policy.
