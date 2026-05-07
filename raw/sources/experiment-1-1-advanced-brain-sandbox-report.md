# Experiment 1.1: The Advanced Brain Sandbox Report

## Purpose

This report summarizes the design, execution, and findings of `Experiment 1.1: The Advanced Brain Sandbox` for the Sound Forge Brain module prototype.

This experiment extends `Experiment 1` from a small flat parameter payload to a larger nested synth patch with explanation-first output and explicit sparse-update semantics.

The goal was to test whether a local LLM can still behave as a deterministic sound-design control surface when the contract becomes more demanding in three ways:

- a larger nested parameter surface
- a required visible `explanation` string as the first key
- sparse updates where unchanged parameters must be emitted as `null`

This is still a control-plane validation step for a future JUCE/C++ implementation rather than an audio-quality evaluation.

## Research Question

Can a local LLM, running under grammar-constrained decoding, reliably generate one strict nested JSON patch object with:

- a leading `explanation` field
- exact block order
- valid numeric ranges
- sparse `null` no-op semantics relative to a provided current patch context

Secondary questions:

- Does the larger nested contract materially increase latency compared with `Experiment 1`?
- Can a 3B-class model preserve sparse delta behavior, or is a 7B-class model required?
- Does visible explanation increase the risk of formatting drift or other control-path failures?

## Parameter Surface Under Test

The experiment used one required top-level JSON object with these exact keys in this exact order:

- `explanation`
- `oscillator-1`
- `oscillator-2`
- `filter`
- `envelope`
- `effects`

The nested synth surface contained `30` leaf parameters total.

### Block structure

- `oscillator-1`: `shape`, `octave`, `semitone`, `detune-cents`, `level`, `unison-voices`
- `oscillator-2`: `shape`, `octave`, `semitone`, `detune-cents`, `level`, `sync-mode`
- `filter`: `mode`, `cutoff-hz`, `resonance`, `drive`, `keytrack`, `env-amount`, `slope-db-per-oct`
- `envelope`: `attack-ms`, `decay-ms`, `sustain`, `release-ms`, `velocity-amount`
- `effects`: `reverb-mix`, `delay-mix`, `chorus-mix`, `stereo-width`, `output-gain-db`, `distortion-drive`

The output contract required:

- one JSON object only
- exact top-level key order
- no extra keys
- explanation as the first field
- explanation length between `20` and `280` characters
- `null` as the only no-op representation
- all nested blocks and all leaf keys always present

## Experimental Architecture

The experiment was implemented in Python using `llama-cpp-python` with GGUF models and a GBNF grammar.

### Constraint strategy

Two layers of enforcement were used.

1. Generation-time constraint
- A GBNF grammar restricted the model to the exact nested JSON structure, exact block order, exact key names, and allowed enum/value forms.

2. Post-parse validation
- Python validation enforced:
  - required keys
  - exact key order
  - numeric finiteness
  - numeric range limits
  - explanation length and presence
  - sparse patch semantics relative to `current_patch_context`

This distinction mattered in practice:

- the grammar solved structure
- the validator solved semantic boundaries and sparse-delta correctness

## Prompting Strategy

The system prompt explicitly instructed the model to:

- emit only one JSON object
- keep `explanation` first
- prefer `null` aggressively for unchanged parameters
- use current patch context when present
- keep the JSON compact

The final prompt and grammar did not attempt to trust model goodwill. They physically constrained the output shape and relied on validation for remaining semantic checks.

## Runtime Environment

The completed experiment used the same host Docker GPU workflow proven in `Experiment 1`.

- shared image: `sound-forge-brain-sandbox:gpu`
- CUDA-enabled `llama-cpp-python`
- repo bind-mounted into the container
- GGUF model directory bind-mounted read-only

Successful runs reported CUDA discovery on the project machine:

- `ggml_cuda_init: found 1 CUDA devices`
- detected GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

This confirms that the authoritative runs used CUDA GPU offload rather than CPU-only inference.

## Models Evaluated

1. `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
2. `Qwen2.5-7B-Instruct-Q4_K_M.gguf`

As in `Experiment 1`, the 3B model was treated as the smaller plumbing and robustness baseline, while the 7B model was the stronger comparison point.

## Fixture Sets

Two fixture sets were used.

### 1. Baseline fixture set

File:

- `Experiments/Experiment 1.1/fixtures/baseline_prompts.json`

This set contained five prompts:

- two full-design prompts without current context
- three sparse-with-context prompts targeting filter, envelope, and effects changes

### 2. Adversarial fixture set

File:

- `Experiments/Experiment 1.1/fixtures/adversarial_prompts.json`

This set contained five robustness-oriented prompts stressing:

- markdown / reasoning pressure
- requests for extra unsupported features
- explicit out-of-range numeric requests
- sparse-pressure wording with current context
- multilingual instructions

## Measurement Method

For every prompt, the experiment recorded:

- raw output
- parsed payload
- syntax validity
- schema validity
- range validity
- key-order validity
- explanation validity
- sparse-patch validity
- finish reason
- prompt/completion token counts
- tokens per second
- per-prompt latency in milliseconds

It also recorded:

- model load time
- parse latency
- validation latency
- sparse extraction latency
- explanation length
- changed parameter count
- null ratio
- all-null no-op count
- failure-category counts

Inference latency was measured with `time.perf_counter()`.

## Operational Note: Token Budget Fix

The first 3B smoke test used `max_tokens = 320` and failed because the model was truncated mid-object.

Artifact:

- `Experiments/Experiment 1.1/results/run_20260504T043704Z.json`

Observed behavior:

- `finish_reason = length`
- JSON was cut off inside the `effects` block
- completion tokens reached exactly `320`

The default generation budget was then increased to `640`, and the system prompt was tightened to prefer compact JSON.

The follow-up smoke test passed:

- `Experiments/Experiment 1.1/results/run_20260504T044116Z.json`

This fix was required before the authoritative benchmark runs.

## Relevant Run Artifacts

### Smoke tests

- `Experiments/Experiment 1.1/results/run_20260504T043704Z.json` - initial 3B smoke test, truncated at `320` tokens
- `Experiments/Experiment 1.1/results/run_20260504T044116Z.json` - 3B smoke test after the `640` token-budget fix

### Baseline full runs

- `Experiments/Experiment 1.1/results/run_20260504T045814Z.json` - Qwen 2.5 3B baseline
- `Experiments/Experiment 1.1/results/run_20260504T045919Z.json` - Qwen 2.5 7B baseline

### Adversarial full runs

- `Experiments/Experiment 1.1/results/run_20260504T050047Z.json` - Qwen 2.5 3B adversarial
- `Experiments/Experiment 1.1/results/run_20260504T050156Z.json` - Qwen 2.5 7B adversarial

### Status logs

- `Experiments/Experiment 1.1/logs/status_qwen3b_baseline.log`
- `Experiments/Experiment 1.1/logs/status_qwen7b_baseline.log`
- `Experiments/Experiment 1.1/logs/status_qwen3b_adversarial.log`
- `Experiments/Experiment 1.1/logs/status_qwen7b_adversarial.log`

## Results

## Baseline Results

### Qwen 2.5 3B Q4_K_M baseline

Artifact:

- `run_20260504T045814Z.json`

Summary:

- total prompts: `5`
- overall pass: `2/5`
- syntax pass: `5/5`
- schema pass: `5/5`
- range pass: `5/5`
- key-order pass: `5/5`
- explanation pass: `5/5`
- sparse pass: `2/5`

Latency:

- model load: `11563.86 ms`
- mean prompt latency: `10243.05 ms`
- median prompt latency: `10032.27 ms`
- p95 prompt latency: `11348.43 ms`
- min/max prompt latency: `9637.58 ms` / `11660.00 ms`

Sparse metrics:

- mean changed params: `28.8`
- mean null ratio: `0.04`

Failure categories:

- `pass`: `2`
- `sparse_fail`: `3`

Observed failures:

- `sparse_brighter_001`
- `sparse_attack_001`
- `sparse_space_001`

Observed behavior:

- The model stayed structurally compliant.
- It failed because it copied unchanged values from `current_patch_context` instead of emitting `null` for no-op parameters.
- In other words, the 3B model handled the nested contract structurally, but not delta-sparsely.

### Qwen 2.5 7B Q4_K_M baseline

Artifact:

- `run_20260504T045919Z.json`

Summary:

- total prompts: `5`
- overall pass: `5/5`
- syntax pass: `5/5`
- schema pass: `5/5`
- range pass: `5/5`
- key-order pass: `5/5`
- explanation pass: `5/5`
- sparse pass: `5/5`

Latency:

- model load: `30523.43 ms`
- mean prompt latency: `10993.83 ms`
- median prompt latency: `9962.44 ms`
- p95 prompt latency: `13144.34 ms`
- min/max prompt latency: `9735.06 ms` / `13449.08 ms`

Sparse metrics:

- mean changed params: `12.0`
- mean null ratio: `0.60`

Interpretation:

- The 7B model was fully stable across the baseline set.
- It handled both full-design and sparse-with-context cases correctly.
- The three sparse baseline cases were meaningfully sparse in practice:
  - `sparse_brighter_001`: `2` changed params, `28` nulls
  - `sparse_attack_001`: `1` changed param, `29` nulls
  - `sparse_space_001`: `3` changed params, `27` nulls

## Adversarial Results

### Qwen 2.5 3B Q4_K_M adversarial

Artifact:

- `run_20260504T050047Z.json`

Summary:

- total prompts: `5`
- overall pass: `3/5`
- syntax pass: `5/5`
- schema pass: `5/5`
- range pass: `4/5`
- key-order pass: `5/5`
- explanation pass: `5/5`
- sparse pass: `4/5`

Latency:

- model load: `15086.76 ms`
- mean prompt latency: `10184.24 ms`
- median prompt latency: `10954.31 ms`
- p95 prompt latency: `11234.25 ms`

Failure categories:

- `pass`: `3`
- `range_fail`: `1`
- `sparse_fail`: `1`

Observed failures:

- `adv_numeric_pressure_001`
  - obeyed the hostile numeric request directly
  - emitted out-of-range values such as `cutoff-hz = 50000`, `attack-ms = 0`, `release-ms = 999999`, `reverb-mix = 2.5`, `stereo-width = 10`
  - validator correctly rejected the payload

- `adv_sparse_pressure_001`
  - ignored the sparse-delta requirement under pressure
  - copied the entire current patch back out with one filter change instead of nulling unchanged fields

Important nuance:

- The 3B model still resisted formatting attacks and extra-key pressure.
- Its failures were semantic: numeric-boundary obedience and sparse-delta obedience.

### Qwen 2.5 7B Q4_K_M adversarial

Artifact:

- `run_20260504T050156Z.json`

Summary:

- total prompts: `5`
- overall pass: `4/5`
- syntax pass: `5/5`
- schema pass: `5/5`
- range pass: `4/5`
- key-order pass: `5/5`
- explanation pass: `5/5`
- sparse pass: `5/5`

Latency:

- model load: `32620.13 ms`
- mean prompt latency: `10666.25 ms`
- median prompt latency: `10318.35 ms`
- p95 prompt latency: `12100.47 ms`

Sparse metrics:

- mean changed params: `3.6`
- mean null ratio: `0.88`
- all-null no-op count: `1`

Failure categories:

- `pass`: `4`
- `range_fail`: `1`

Observed failure:

- `adv_numeric_pressure_001`
  - still emitted illegal values under explicit hostile numeric pressure
  - returned only the pressured fields as changes, but did not clamp them into legal ranges

Important nuance:

- In `Experiment 1`, the 7B model had resisted the adversarial numeric-pressure case.
- In `Experiment 1.1`, the richer nested sparse contract weakened that robustness.
- This is one of the most important findings of the experiment.

## Adversarial Prompt Behavior Observations

### Markdown / reasoning pressure

Prompts explicitly asked for markdown bullets and reasoning before the patch.

Observed behavior:

- both models still returned one valid JSON object only
- no free text escaped outside the JSON object
- explanation style spillover occurred inside the explanation string itself

Interpretation:

- grammar constraints remained effective at protecting the machine interface
- explanation content can still absorb stylistic prompt contamination even when structure remains correct

### Extra-key pressure

Prompts requested unsupported concepts such as:

- resonance modulation
- tempo sync
- wavetable index
- LFO rate

Observed behavior:

- neither model emitted extra schema keys
- 3B responded with a full legal patch
- 7B responded with an all-null no-op patch and a legal explanation

Interpretation:

- ontology expansion was successfully blocked by the grammar
- 7B showed one useful refusal pattern: preserve the contract and emit no legal change when unsupported requests cannot be represented

### Explicit illegal numeric pressure

Observed behavior:

- both models failed the numeric-pressure prompt in `Experiment 1.1`
- both remained structurally disciplined
- both violated semantic numeric limits

Interpretation:

- post-parse numeric validation remains mandatory
- the larger explanation-first sparse contract made semantic robustness harder than in `Experiment 1`

### Sparse-pressure behavior

Observed behavior:

- 3B failed sparse-pressure again, copying unchanged values from the current patch
- 7B remained sparse and only changed the necessary filter parameters

Interpretation:

- the 3B model is not dependable for delta patches when current-state conditioning is added
- the 7B model is substantially better at representing sparse edits instead of full-state restatement

## Latency Comparison Against Experiment 1

Using the baseline means from the authoritative runs:

- `Experiment 1` 3B baseline mean latency: `1494.22 ms`
- `Experiment 1.1` 3B baseline mean latency: `10243.05 ms`
- latency multiplier: about `6.86x`

- `Experiment 1` 7B baseline mean latency: `2123.34 ms`
- `Experiment 1.1` 7B baseline mean latency: `10993.83 ms`
- latency multiplier: about `5.18x`

Interpretation:

- The larger nested output and required explanation dramatically increased generation latency.
- Parse, validation, and sparse extraction overhead remained negligible; the added cost was overwhelmingly model generation time.

## Comparative Interpretation

## What has been convincingly demonstrated

This experiment supports the following conclusions:

1. A local LLM can emit explanation-first nested synth patches under grammar constraint without drifting into malformed output.
2. A visible `explanation` field can coexist with machine-consumable JSON when the structure is grammar constrained.
3. A 7B-class local model can reliably produce sparse delta patches relative to current patch context for the tested baseline prompts.
4. Consumer NVIDIA hardware can run this richer control path locally, but only with much higher latency than the simpler `Experiment 1` surface.
5. Grammar alone is still not enough; sparse validation and numeric validation after parsing are required.

## What has not been fully demonstrated

The experiment does not yet prove:

1. that the generated parameter values are acoustically optimal
2. that explanation content is consistently useful for a product UI
3. that the behavior will generalize across a much larger prompt corpus
4. that the same behavior will hold identically in a future embedded C++ host
5. that 7B will remain range-robust once the schema grows further

## Why the results matter

The experiment clears a more demanding control-plane question than `Experiment 1`.

The project now has evidence that a local Brain can emit a larger nested structured patch with visible reasoning and sparse semantics. However, it also shows that increasing contract complexity materially changes model robustness.

The key shift is:

- structure stayed robust
- sparse semantics became model-quality sensitive
- numeric-boundary obedience became harder even for the stronger model

## Limitations

The following limitations should be noted:

1. Small fixture count
- baseline set: `5` prompts
- adversarial set: `5` prompts

2. No acoustic scoring rubric
- outputs were validated structurally, numerically, and sparsely
- no formal human evaluation of patch plausibility was applied

3. Explanation quality not separately scored
- explanation length and presence were enforced
- usefulness and UI value were not evaluated

4. Python host implementation
- this remains a prototype host, not the final JUCE/C++ environment

5. Range behavior under adversarial pressure remains weak
- even the 7B model failed the numeric-pressure case in this richer contract

## Practical Implications for Sound Forge

1. Explanation-first output is viable, but it increases token cost significantly.

2. Sparse nested patching should not use a 3B-class model if correct delta behavior matters.

3. Even with a 7B-class model, the plugin must validate and possibly clamp numeric values after parsing.

4. This inference path is even more clearly a background-thread control-plane task, not an audio-thread candidate.

5. The system should continue to treat grammar, sparse validation, range validation, and downstream DSP-safe application as separate contracts.

## Recommended Next Experiments

### 1. Sparse-prompt refinement pass

Test whether a tighter prompt or few-shot examples materially improve 3B sparse-delta behavior, or whether the limitation is fundamentally model capacity.

### 2. Clamp-versus-reject in the nested sparse setting

Compare strict rejection with safe post-parse clamping for numeric-pressure failures.

### 3. Explanation usefulness evaluation

Score whether explanations are merely valid strings or actually useful to a UI, logging, or user-trust layer.

### 4. Larger sparse/adversarial corpus

Expand the fixture library with:

- more no-op cases
- more mixed-intent sparse edits
- more multilingual sparse prompts
- more unsupported-feature prompts
- more adversarial numeric requests

### 5. Stronger-model comparison

Evaluate whether a stronger quant or model tier restores numeric robustness without unacceptable extra latency.

## Final Conclusion

Experiment 1.1 successfully demonstrated that the Sound Forge Brain concept can be extended from a small flat parameter payload to a richer explanation-first nested sparse patch contract.

The strongest validated conclusion is:

> A local LLM can emit deterministic, schema-compliant, explanation-first nested JSON synth patches with sparse `null` semantics under grammar constraint, but sparse-delta fidelity and numeric-boundary obedience become significantly more model-sensitive as the contract grows.

The most important comparative findings are:

- 7B remained reliable for baseline sparse patch generation
- 3B did not remain reliable for sparse-with-context generation
- both models still required post-parse range enforcement under adversarial numeric pressure
- generation latency increased by about `6.86x` for 3B and `5.18x` for 7B relative to `Experiment 1`

For current project direction, the evidence supports the following position:

- keep grammar-constrained decoding
- keep strict post-parse validation
- treat sparse validation as a first-class requirement
- keep inference off the audio thread
- prefer the stronger 7B model when sparse delta behavior matters

The remaining open questions are now less about whether the Brain can stay machine-structured, and more about numeric safety policy, semantic patch quality, explanation usefulness, and eventual C++ deployment behavior.
