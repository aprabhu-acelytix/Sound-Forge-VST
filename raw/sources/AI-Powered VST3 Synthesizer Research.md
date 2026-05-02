# **Architecting an Agentic VST3 Synthesizer: A Technical Analysis of Real-Time AI and C++ DSP Integration**

The intersection of generative artificial intelligence and real-time digital signal processing (DSP) represents a profound paradigm shift in audio software architecture. Designing a VST3 hybrid synthesizer equipped with an onboard "System of Models"—an agentic workflow comprising an "Ear" for audio feature extraction, a "Brain" for intelligent parameter mapping, and a "Tutor" for conversational interactions—presents unprecedented technical challenges. Traditional audio plugin development demands rigid adherence to real-time safety constraints, strict memory management, and highly optimized CPU utilization. Conversely, large language models (LLMs) and neural audio classifiers are notoriously non-deterministic in their execution times, highly memory-intensive, and inherently prone to blocking operations.

Bridging this gap requires a sophisticated C++ architecture within the JUCE framework. The system must completely isolate neural network inference from the critical audio callback, utilize highly optimized lightweight models, enforce strict output schemas using formal grammars, and maintain a seamless integration with the host application's parameter state. The ensuing analysis provides an exhaustive technical deep dive into building this agentic synthesizer, dissecting the current state-of-the-art (SOTA) methodologies, open-source libraries, and architectural patterns required to deploy local AI models effectively without compromising DSP integrity.

## **1\. Real-Time AI Integration in C++: Resolving the Computational Bottleneck**

The primary engineering hurdle in integrating machine learning into a VST3 plugin is the fundamental incompatibility between the operational requirements of neural network inference and the constraints of the real-time audio thread. Digital Audio Workstations (DAWs) rely on deterministic execution to ensure continuous audio playback.

### **1.1 The Golden Rule of Audio Programming and Priority Inversion**

In JUCE and similar audio frameworks, the audio callback function (the processBlock method) operates on a high-priority, real-time thread assigned by the host operating system or DAW.1 The "Golden Rule" of audio programming dictates that a developer must never allocate memory dynamically, interact with the file system, acquire a mutex or lock, or execute unbounded operations on this specific thread.2 Failing to adhere to this rule results in thread preemption or priority inversion. If the audio thread is forced to wait for an unbounded operation—such as a neural network forward pass that allocates memory via malloc—it will miss its processing deadline, resulting in audio dropouts, zipper noise, and severe host instability.3

Machine learning inference engines, by their general-purpose nature, routinely violate these constraints. Operations involving deep neural networks typically require dynamic heap allocations for massive tensors, extensive matrix multiplications, and complex thread-pool synchronizations.4 Therefore, direct execution of LLM text generation or complex audio-embedding extraction within the processBlock is strictly prohibited. The architecture must adopt an asynchronous, multi-threaded model where DSP and AI operate in entirely decoupled environments.6

### **1.2 Thread Separation and Lock-Free Synchronization Strategies**

To safely integrate an agentic workflow, the plugin architecture necessitates strict thread boundaries. The system will consist of three primary execution contexts:

1. **The Audio Thread:** Handles the DSP (synthesis generation inside processBlock), writes incoming audio to a lock-free ring buffer for the "Ear" to analyze, and reads target parameter states via atomic variables.8  
2. **The Worker Thread Pool:** Reads from the audio ring buffer, executes feature extraction (the "Ear"), feeds data to the LLM inference engine or HTTP client (the "Brain"), and parses the resulting output.6  
3. **The Message (GUI) Thread:** Updates the conversational UI, reflects changes on visual knobs, and manages the AudioProcessorValueTreeState (APVTS) synchronization.10

Communication between the Audio Thread and the Worker Thread must utilize a Single-Producer, Single-Consumer (SPSC) lock-free FIFO queue, such as juce::AbstractFifo.10 The audio thread pushes continuous samples into this FIFO. Once a sufficient block size is reached (e.g., 2048 or 4096 samples), the background worker thread consumes this block.8 This pattern prevents any OS-level locking or CPU blocking. For optimal performance, the implementation should leverage C++ atomics (std::atomic) with explicit memory ordering (std::memory\_order\_release when pushing data, and std::memory\_order\_acquire when popping data) to ensure visibility across CPU cores without the overhead of heavy synchronization primitives.2 The use of Compare-And-Swap (CAS) loops or Read-Copy-Update (RCU) mechanisms should generally be avoided for continuous audio streaming due to their potential to introduce unbounded wait states under heavy contention.10

### **1.3 Local LLM Inference Engines: llama.cpp Integration**

For the "Brain" component of the agentic workflow, llama.cpp has emerged as the definitive industry standard for local, C++ based inference of large language models.13 Written in pure C/C++ with zero dependencies, it supports advanced hardware acceleration frameworks including Apple Metal, NVIDIA CUDA, and Vulkan.14 It heavily utilizes integer quantization (specifically the GGUF format), making it viable to run highly capable models (such as Qwen-2.5-Coder or Llama-3 8B) on consumer hardware with limited VRAM.14

Integrating llama.cpp directly into a JUCE VST3 plugin allows the application to remain entirely self-contained. A notable example of this architecture in practice is the open-source LLMidi plugin, which embeds a llama.cpp backend to generate MIDI patterns locally without requiring internet access.16 When building such a system, developers must configure the CMake build pipeline to compile llama.cpp alongside JUCE. The llama\_context and llama\_model instances must be loaded on a background thread to prevent stalling the DAW during the heavy disk I/O phase of reading a multi-gigabyte GGUF file.17

Furthermore, the configuration of llama.cpp parameters is crucial for a plugin environment. Setting appropriate context window limits (n\_ctx) and utilizing batch processing for prompt ingestion ensures that the plugin does not exhaust the host machine's RAM.18 Because LLM inference is highly CPU and GPU intensive, the background worker thread executing llama\_decode must run at a lower priority than the DAW's audio thread to prevent CPU starvation.

### **1.4 Out-of-Process Execution via Modern C++ HTTP Bridging**

Given the massive resource consumption of an LLM, embedding a model directly into a plugin's memory space can cause DAW instability, high memory pressure, and thermal throttling, particularly if the user is running multiple instances of the synthesizer.15 A highly effective alternative architectural pattern is out-of-process execution using a local API server.20

By requiring the user to run a local server like Ollama, LM Studio, or a standalone llama-server process, the VST3 plugin acts solely as a lightweight client.13 This approach completely isolates the heavy AI computation from the DAW process. Integration in C++ can be achieved using modern libraries such as ollama-hpp, a header-only C++ binding that interacts directly with the Ollama REST API.21 The plugin communicates with the local host (typically http://localhost:11434) via HTTP POST requests containing the prompt and desired model parameters.20

To maintain real-time safety, all HTTP requests must be executed asynchronously. In JUCE, this requires utilizing juce::Thread or juce::ThreadPool to spin up a worker thread that dispatches the juce::URL or underlying curl request, awaits the response, and subsequently passes the serialized JSON string to the message thread via a lock-free queue or a juce::AsyncUpdater.23 This ensures the DAW's user interface and audio engine remain perfectly responsive while the external server generates the response over several seconds.

### **1.5 Evaluating ONNX Runtime for Feature Extraction**

While llama.cpp powers the "Brain," the "Ear" component—responsible for neural audio feature extraction or audio tagging—requires a different ecosystem. Microsoft's ONNX Runtime (ORT) provides a performant solution to inference models from various source frameworks (PyTorch, TensorFlow) using a unified C++ API.24 ORT supports a vast array of hardware accelerators (CUDA, CoreML, DirectML) through its Execution Providers (EP) framework.25

However, ORT is fundamentally not designed for real-time applications.4 During profiling, ORT exhibits intermittent heap allocations (malloc calls) and relies on internal thread pools.4 When deploying ORT in an audio plugin, execution must absolutely be delegated to a background thread pool.7

The threading configuration of ORT is a critical consideration for VST development. ORT utilizes intra\_op\_num\_threads to parallelize computation inside each operator, and inter\_op\_num\_threads to parallelize across nodes.28 By default, ORT may enable "Thread-Pool Spinning," where worker threads spin in a busy-wait loop anticipating new work.28 While this provides faster inference, it consumes massive amounts of CPU cycles and power, which is disastrous in a DAW environment where multiple audio tracks are competing for CPU time. Developers must explicitly disable this spinning behavior by modifying the session options (e.g., setting session.intra\_op.allow\_spinning to "0").28

Integrating ORT into a VST3 also introduces complexities regarding static versus dynamic linking. Due to potential protobuf dependency clashes and binary bloat, industry best practice involves compiling customized, dynamically linked ORT libraries with vendor-specific prefixes, deployed to specific system directories (e.g., System32 on Windows or the bundle Frameworks path on macOS) to avoid conflicts with other plugins.26

### **1.6 Micro-Neural DSP: Real-Time Inline Execution**

If the agentic workflow involves directly modifying an inline neural DSP component (such as a neural waveshaper, amp simulation, or physical modeling non-linearity), a background thread is insufficient due to the required latency.7 In these specific cases, libraries designed expressly for the audio thread must be employed.

RTNeural is a C++ library designed for real-time inference of Neural Network models with audio in mind.31 It guarantees zero allocations and lock-free execution after initialization, allowing it to be called directly within the processBlock.6 Academic benchmarking reveals that while ONNX Runtime may exhibit the lowest average runtimes for stateless models, it suffers from real-time violations (latency spikes) under strict audio constraints.7 In contrast, RTNeural is specifically tuned for minimum worst-case execution times, ensuring deterministic behavior.27 While RTNeural cannot run massive generative models or large audio classifiers, it is the premier choice for micro-scale neural networks operating directly on the signal path.

| Inference Engine | Primary Use Case | Real-Time Safe? | CPU/Memory Footprint | Architectural Role |
| :---- | :---- | :---- | :---- | :---- |
| **ONNX Runtime** | Audio feature extraction, CNNs, Audio Tagging | No (requires background thread) | High (requires tuned thread configs) | Background "Ear" analysis |
| **llama.cpp** | Local LLM text generation, JSON payload creation | No (requires background thread) | Very High (requires GGUF quantization) | Background "Brain" processing |
| **RTNeural** | Micro-neural DSP, distortion, waveshaping | Yes (zero allocations post-init) | Very Low | Inline audio processBlock |
| **Ollama / REST API** | Out-of-process LLM server execution | N/A (Async HTTP request) | External to Plugin Process | Separated "Brain" via Local API |

## **2\. The "Ear": Audio-to-Text and Feature Extraction**

The "Ear" serves as the perception layer of the agentic workflow. Its objective is to analyze the incoming audio signal or the current synthesizer output and translate the sonic characteristics into a structured, semantic representation that the LLM "Brain" can interpret. This requires bridging the gap between raw time-domain waveforms and high-level descriptive text.

### **2.1 Algorithmic DSP Analysis via Essentia**

The extraction of audio features can be approached through classical algorithmic DSP or modern neural embeddings. For highly deterministic, lightweight, and mathematically precise measurements of synthesizer timbres, classical algorithmic analysis is vastly superior in a real-time context.

The open-source Essentia C++ library provides an extensive, highly optimized collection of algorithms designed specifically for Music Information Retrieval (MIR) and audio analysis.34 Released under the Affero GPLv3 license, it calculates robust time-domain and spectral characteristics—such as Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, zero-crossing rate (ZCR), spectral kurtosis, and transient responses.36

The calculation of these features involves complex mathematical transformations. For instance, computing MFCCs begins with converting audio signals from the time domain to the frequency domain via a Fast Fourier Transform (FFT).37 The Mel scale is then employed through Mel filters that mimic the human ear's logarithmic frequency perception. Finally, taking the logarithm of the energies in each Mel filter and applying a Discrete Cosine Transform (DCT) yields the MFCCs, which provide a highly compact representation of the spectral envelope.37 The spectral centroid, representing the "brightness" of a sound, requires computing the magnitude of each frequency bin and calculating the weighted average of the frequencies.39

Essentia is uniquely positioned for VST integration because it features a "streaming mode" explicitly designed for real-time audio networks.40 Algorithms can be connected via a RingBufferInput, allowing the background worker thread to continuously pull data from the audio thread's lock-free FIFO and process it dynamically.40 Unlike Python-based counterparts (like Librosa) which are unsuitable for plugins, Essentia's C++ core is optimized for computational efficiency and memory constraints.42 By extracting numeric data (e.g., "Spectral Centroid: 4500Hz, Attack Time: 12ms"), the "Ear" can pass a compact, dense JSON summary to the LLM, informing the "Brain" of the exact physical state of the sound.38

### **2.2 Neural Audio Tagging and Lightweight Classification**

While algorithmic data is precise, it lacks high-level semantic understanding. An LLM may struggle to interpret an array of raw MFCCs, but it perfectly understands semantic descriptors like "warm," "distorted," or "plucked." To provide this context, the "Ear" must employ lightweight neural classification.

The sherpa-onnx library provides excellent C++ APIs for deploying pre-trained audio tagging models.43 Utilizing models like the Zipformer audio tagger, the system can classify short segments of audio into predefined categories (e.g., identifying instrument types or transient characteristics) without relying on massive neural networks.43 These models can be exported to ONNX format and run efficiently on CPU via the background thread pool, translating the raw waveform into an array of confidence-weighted semantic tags.43

### **2.3 Multimodal Semantic Embeddings: CLAP and TRR**

Advanced semantic understanding requires multi-modal models that map audio and text into a shared latent space. CLAP (Contrastive Language-Audio Pretraining) achieves exactly this.47 Using CLAP, the "Ear" can convert an audio buffer into a dense vector embedding, which can then be used to retrieve textual captions or search a database of synthesizer presets.48 Research such as the *SynthScribe* project demonstrates the profound efficacy of using LAION-CLAP embeddings to allow musicians to retrieve and organically modify synthesizer sounds via text and audio queries.49 *SynthScribe* implements features that address searching through sounds and creating completely new sounds using genetic algorithms driven by these multimodal embeddings.50

However, CLAP is primarily designed for broad classification and captioning tasks, which may not capture the fine-grained nuances of synthesizer modulations. The DAFx 2024 paper *TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control* introduces a more specialized approach for timbral analysis.51 *TimberAgent* utilizes mid-level Wav2Vec2 representations to capture "timbral texture" for parameter retrieval.51 Instead of relying on single-vector embeddings, it calculates the Gram matrices of projected mid-level activations, which preserves the texture-relevant co-activation structure (second-order statistics) of the audio signal.51 This "Texture Resonance Retrieval" (TRR) allows the system to accurately match the texture of a target sound, yielding the strongest parameter alignment among evaluated retrieval baselines, outperforming standard CLAP retrieval for complex audio effects.52 Integrating TRR calculations in C++ allows the "Ear" to mathematically summarize complex textures for the agentic workflow.

### **2.4 Large Audio-Language Models**

Recently, multimodal LLMs like Qwen2-Audio and Qwen3-Omni have demonstrated state-of-the-art audio captioning capabilities.53 These models adopt a Thinker–Talker Mixture-of-Experts architecture that can directly ingest audio tokens and output highly detailed, granular technical summaries of a timbre.53 For instance, a model could output: "The audio features a heavily detuned sawtooth wave with a fast filter envelope and extensive comb filtering."

However, the parameter count of these models (e.g., Qwen3-Omni-30B-A3B-Captioner) prohibits native real-time C++ integration in a standard desktop VST3.55 To utilize such advanced models, the plugin architecture must strictly employ the out-of-process HTTP bridging strategy (discussed in Section 1.4), sending compressed audio snippets to a local or cloud-based API server for captioning.

### **2.5 Implementing the "Ear" Architecture**

For a robust hybrid VST3, the optimal approach combines algorithmic efficiency with neural semantics:

1. **Continuous Fast Path (Essentia C++):** The background worker thread continuously computes MFCCs and spectral envelopes via Essentia, maintaining a rolling JSON state of the timbre.34  
2. **Triggered Slow Path (ONNX/Neural):** Upon a specific user request to the "Tutor," a larger buffer of audio (e.g., 2 seconds) is captured from the lock-free queue. This buffer is processed through a lightweight ONNX model (like sherpa-onnx) or a TRR Gram matrix calculation. The resulting semantic tags are appended to the Essentia JSON data.  
3. **Payload Generation:** The combined numeric and semantic data is formulated into a dense contextual string, serving as the system prompt for the "Brain."

## **3\. The "Brain": LLM Parameter Mapping and APVTS Integration**

The "Brain" of the agentic workflow is tasked with receiving natural language prompts from the user, combining them with the sonic context provided by the "Ear," and deducing the exact matrix of synthesizer parameters required to achieve the desired sound. To function within a C++ audio plugin, the output must be deterministic, strictly formatted, and safely applied to the audio engine.

### **3.1 Prompt Engineering and The Ambiguity Tax**

In traditional text-to-text generation, LLMs produce free-form prose. In a software architecture, free-form outputs generate "parsing nightmares" and brittle regular expression dependencies, leading to high failure rates when mapping to exact software parameters.56 To operate as a programmatic "Copilot," the LLM must behave as a data processor that adheres to strict schema definitions.57 The operational cost of relying on vague, unpredictable inputs in workflows that demand machine-level precision is known as the "Ambiguity Tax".56

Eliminating this tax requires advanced prompt engineering. Prompt engineering involves designing structured sequences of instructions, exemplar data, and domain knowledge to guide the model's generative behavior.58 For a synthesizer, the prompt must assign a specific persona ("Expert Sound Designer") and provide a precise JSON schema that represents every available parameter.57

The prompt must explicitly define parameter boundaries and enumerations to constrain the model's output.60 For example, the prompt must include: "filter\_cutoff": { "type": "number", "minimum": 20.0, "maximum": 20000.0, "description": "Lowpass filter cutoff frequency in Hz." } "oscillator\_type": { "type": "string", "enum": \["sine", "sawtooth", "square", "triangle"\] }

Furthermore, utilizing techniques like Few-Shot Prompting and Chain-of-Thought (CoT) prompting significantly improves performance. By providing the model with a few input-output examples of how acoustic requests map to parameter values, and forcing the model to generate a \<think\> block explaining its reasoning before outputting the final JSON, the system can achieve dramatic increases in accuracy and complex task completion.61

### **3.2 Grammar-Constrained Decoding (GBNF)**

Even with exemplary prompt engineering, LLMs can occasionally hallucinate formatting or append conversational filler (e.g., "Here is the JSON you requested:..."). To guarantee 100% schema adherence for synthesizer control, the inference engine must enforce grammar-constrained decoding.

If using llama.cpp for native execution, the framework supports GGML BNF (GBNF) grammars.63 By supplying a .gbnf file that rigidly defines the syntactic structure of a valid JSON object, the inference engine masks out invalid tokens at the logits level during generation.65

A GBNF rule defines how a non-terminal can be replaced with sequences of terminals (Unicode code points).63 For example, a rule enforcing a simple JSON object might look like: root ::= "{" ws "\\"cutoff\\"" ws ":" ws number ws "}" This constraints the model to *only* generate valid JSON conforming to the defined structure. By enforcing this at the C++ inference level, the system guarantees that the VST parameter parsing logic will never crash due to malformed string outputs, completely transforming the LLM into a reliable, machine-readable component.56

### **3.3 JUCE APVTS Integration and Lock-Free State Management**

Once the background worker thread successfully generates and parses the JSON payload, the new parameter values must be applied to the synthesizer. In modern JUCE architecture, plugin state, host automation, and GUI synchronization are managed by the AudioProcessorValueTreeState (APVTS).67 The APVTS acts as the central data structure, wrapping a ValueTree and maintaining a collection of AudioProcessorParameter objects.67

Integrating a background LLM thread with the APVTS requires meticulous thread synchronization to avoid violating the Golden Rule of audio programming.

**The Audio Thread Reading Mechanism:** Reading parameters from the APVTS on the audio thread is natively thread-safe. APVTS::getRawParameterValue() returns a std::atomic\<float\>\*.10 The audio processBlock simply dereferences these atomic pointers to obtain the current parameter values lock-free, ensuring no blocking occurs during DSP execution.11

**The Writing Bottleneck and Thread Conflicts:** However, *writing* to the APVTS to apply the LLM's JSON payload is a highly complex operation. Updating a parameter programmatically requires calling setValueNotifyingHost() or interacting with the ValueTree directly. Internally, functions that notify listeners of parameter changes (such as AudioProcessorParameter::sendValueChangedMessageToListeners) acquire a ScopedLock (specifically, listenerLock).70 This lock is necessary to avoid undefined behavior or crashes if the listener list is modified while the system is iterating through it to broadcast the change.70

Because of this internal mutex, it is strictly forbidden to call parameter update functions from the real-time audio thread.10 Furthermore, if the background worker thread (which parses the LLM output) calls this directly, it risks severe race conditions with the Message (GUI) thread, potentially causing UI glitches, deadlocks, or host DAW instability.70

**Architectural Best Practice for Agentic Parameter Updates:**

The only safe and compliant methodology is to route the JSON parameter payload from the LLM Worker Thread to the main GUI/Message Thread before touching the APVTS. This is achieved through the following pattern:

1. **Generation & Parsing:** The LLM Worker thread generates the JSON, validates it against the schema, and extracts the key-value pairs (e.g., {"cutoff": 1200.0, "resonance": 0.7}).  
2. **Lock-Free Queuing:** The worker thread places these parsed structures into a thread-safe, lock-free message queue (e.g., a custom juce::AbstractFifo implementation or juce::ConcurrentLinearFIFO) designed for cross-thread communication.10  
3. **Message Thread Dispatching:** A juce::AsyncUpdater or a high-frequency juce::Timer running exclusively on the main Message Thread continuously polls this queue.71  
4. **Parameter Application:** When a new parameter payload is detected, the Message Thread pops the data and iterates through the key-value pairs. For each parameter, the Message Thread initiates the change by calling beginParameterChangeGesture(), followed by setValueNotifyingHost(), and finally endParameterChangeGesture().72

This architecture guarantees several critical outcomes: the DAW's automation lanes record the AI's adjustments as proper user interactions, the GUI knobs update visually in real-time, the internal APVTS ScopedLock operations are safely handled entirely within the correct Message Thread context, and the audio thread continuously receives the new values totally lock-free via its atomic pointers.10

| Component Thread | Action | Synchronization Mechanism | Real-Time Safe? |
| :---- | :---- | :---- | :---- |
| **LLM Worker** | Generates JSON, parses values, pushes to Queue | std::memory\_order\_release | N/A (Background) |
| **Message (GUI)** | Pops Queue, calls setValueNotifyingHost | juce::AsyncUpdater \+ ScopedLock | N/A (Main Thread) |
| **Audio Thread** | Reads getRawParameterValue() | std::atomic\<float\> load | Yes (Lock-Free) |

## **4\. Hybrid Synthesizer Architecture: Building the DSP Engine**

With the robust agentic infrastructure established, the foundational DSP engine must be capable of translating the LLM's vast parameter arrays into high-fidelity sound. A modern hybrid synthesizer typically combines Wavetable synthesis—for complex, evolving, and mathematically precise timbres—with Physical Modeling—for organic, acoustic realism and chaotic non-linearities.73 Building this from the ground up requires leveraging highly optimized C++ DSP libraries.

### **4.1 Modern C++ DSP Libraries for JUCE**

Developing a robust synthesis engine from scratch is notoriously error-prone, particularly regarding digital aliasing, stability of recursive filters, and cache optimization. Several open-source C++ DSP libraries exist to accelerate this process, each with varying degrees of suitability for a commercial VST3.

**JUCE DSP Module (juce\_dsp):** The native JUCE DSP module provides highly optimized, template-driven C++ classes for processing blocks of audio. It includes state-variable filters, waveshapers, fast math approximations, and basic oscillators.75 Because it is deeply integrated into the JUCE framework, it utilizes SIMD (Single Instruction, Multiple Data) vectorization automatically across platforms and ensures maximum cache coherency. However, its built-in oscillator classes are relatively rudimentary and lack advanced wavetable morphing matrices or physical modeling paradigms.

**Maximilian:** Maximilian is a widely cited C++ audio library designed for ease of use, featuring syntax inspired by the Processing environment. It is heavily utilized in academic, live coding, and instructional contexts.75 While it provides a vast array of features—including granular synthesis, equal-power spatialization, and FFT processing—its core oscillators lack the rigorous anti-aliasing techniques (e.g., band-limited interpolation) required for a commercial VST3.78 Naive waveform generation in Maximilian produces significant aliasing artifacts at higher frequencies, making it generally unsuitable for high-end production environments without significant modification.78

**DaisySP:** Developed by Electro-Smith for the Daisy embedded hardware ecosystem, DaisySP is a phenomenal, lightweight C++ DSP library that can be seamlessly compiled into JUCE desktop plugins via CMake.79 It features an expansive collection of production-ready modules, many of which are highly optimized C++ ports of Mutable Instruments' renowned Eurorack hardware modules.80 Crucially, DaisySP includes sophisticated physical modeling algorithms (Karplus-Strong string synthesis, modal synthesis, resonators), advanced drum synthesis models, dynamics processors, and granular players.80 Because it was designed for embedded microcontrollers, its algorithms are highly efficient, though developers must be mindful of allocating buffers appropriately for block-based processing in a VST context.81 DaisySP represents the optimal foundational library for the physical modeling arm of the hybrid synthesizer.

| Library | Key Strengths | Core Limitations | Optimal Role in Hybrid VST |
| :---- | :---- | :---- | :---- |
| **JUCE DSP** | SIMD optimization, perfect JUCE integration, high-quality IIR/FIR filters. | Limited synthesis algorithms, basic oscillators. | Global routing, master effects, filters, oversampling. |
| **DaisySP** | Exceptional physical modeling, Eurorack-style modules, highly efficient. | Originally tuned for embedded hardware limits. | Core generation for strings, resonators, and percussion. |
| **Maximilian** | Easy syntax, extensive feature set, granular synthesis. | High aliasing in naive oscillators, older architecture. | Prototyping, pedagogical experiments, MIR tools. |

### **4.2 State-of-the-Art Wavetable Synthesis and Anti-Aliasing**

Wavetable synthesis involves scanning through a pre-calculated array of single-cycle waveforms to generate periodic oscillations.73 It allows for highly complex harmonic evolutions by morphing between different tables in a three-dimensional matrix.

A major technical challenge in wavetable architecture is digital aliasing. When the LLM Agent commands the synthesizer to jump to very high pitches or apply extreme frequency modulation (FM), the upper harmonics of the wavetable can easily exceed the Nyquist limit (half the sampling rate). These frequencies cannot be represented digitally and fold back into the audible spectrum as dissonant, metallic, inharmonic artifacts.82

To achieve state-of-the-art wavetable performance, the engine must implement rigorous anti-aliasing architectures. A common and highly effective architectural pattern for wavetables is **Mipmapping** combined with phase accumulation and interpolation.83

1. **Mipmapping Generation:** During the plugin's initialization phase (or when a new wavetable is loaded), the system computes multiple versions of each single-cycle waveform using Fast Fourier Transforms (FFT). For each progressively higher octave, the upper harmonics (spectral bins) that would exceed the Nyquist limit at that specific pitch are zeroed out in the frequency domain. An Inverse FFT (IFFT) then converts the data back to the time domain, creating a "band-limited" version of the wave.83  
2. **Phase Accumulation and Interpolation:** The audio thread utilizes a high-precision phase accumulator to index the table. Depending on the current fundamental frequency of the MIDI note, the DSP engine selects the appropriate mipmap level. To prevent audible switching between mipmap levels across the keyboard, the engine dynamically crossfades (interpolates) between the two closest mipmap tables, ensuring a pristine, alias-free signal regardless of the LLM's extreme parameter modulations.83 Alternative SOTA techniques, such as Band-Limited Impulse Train (BLIT) integration or Chebychev polynomial fitting, can also be employed for mathematically generated waveshapes, but mipmapping remains the standard for arbitrary imported audio samples.85

### **4.3 Physical Modeling Engines and Agentic Control**

For the physical modeling component of the hybrid engine, the architecture relies on mathematical simulations of the acoustic properties of real-world instruments. This involves constructing complex networks of delay lines, noise bursts, mass-spring interactions, and specialized filters to simulate exciters (e.g., a bow, a hammer) and resonators (e.g., a string, a membrane, a tube).74 Using DaisySP's robust modal and resonator classes, or porting partial differential equation (PDE) solvers, the synthesizer can generate highly organic, evolving timbres.80

However, physical models (such as waveguides) are notoriously sensitive to their input parameters. A minute shift in the stiffness of a virtual material, the mass of a virtual hammer, or the feedback coefficient of a delay line can cause the entire mathematical model to diverge, self-oscillate, and output extreme, speaker-damaging volumes.86 Designing compelling sounds manually in these environments is often an exercise in frustration for end-users, as the parameter space is chaotic and non-linear.

This is precisely where the agentic "System of Models" becomes an invaluable architectural necessity rather than a mere novelty. By mapping semantic textual concepts (e.g., "Generate a sound like a giant cello struck by a glass hammer, but keep the resonance controlled") through the LLM "Brain," the Copilot can navigate these highly unstable mathematical parameter spaces safely. The LLM, constrained by its JSON schema boundaries and guided by carefully crafted prompts, acts as an intelligent governor. It shields the user from audio blowouts while uncovering complex sweet spots in the acoustic simulation that would be nearly impossible to dial in manually via a traditional GUI.

## **5\. Conclusion**

Building an agentic VST3 hybrid synthesizer in JUCE represents the absolute bleeding edge of audio software development. It requires an architecture defined by strict boundaries, asynchronous orchestration, and meticulous thread safety. The system must never compromise the real-time audio thread, enforcing the golden rule of zero allocations and lock-free data transfer to maintain DSP integrity.

The architecture succeeds by dividing labor into highly specialized domains: The "Ear" operates as an algorithmic and neural observer, utilizing optimized C++ libraries like Essentia and background ONNX inference (such as sherpa-onnx or TRR models) to stream dense semantic audio data without blocking DSP execution. The "Brain" leverages out-of-process REST APIs via ollama-hpp or embedded GGUF inference via llama.cpp, employing strict GBNF grammars to ensure the LLM output behaves as a deterministic, programmatic payload rather than unstructured conversational prose.

This resulting JSON payload is subsequently shuttled through lock-free SPSC queues to the Message Thread, safely automating the complex matrix of JUCE's APVTS via juce::AsyncUpdater and beginParameterChangeGesture. Finally, this safely updated parameter state drives a robust hybrid DSP backend, fusing the alias-free precision of wavetable mipmapping with the organic, algorithmic richness of DaisySP's physical modeling algorithms. By meticulously decoupling neural generation from critical audio computation, this architecture allows a "System of Models" to act as an active, highly intelligent copilot—translating abstract sonic intentions into precise, high-fidelity synthesizer configurations in real-time.

#### **Works cited**

1. Real time thread in Juce \- Audio Plugins \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/real-time-thread-in-juce/43361](https://forum.juce.com/t/real-time-thread-in-juce/43361)  
2. Lock free & real time stuffs (for dummies) \- Getting Started \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/lock-free-real-time-stuffs-for-dummies/58870](https://forum.juce.com/t/lock-free-real-time-stuffs-for-dummies/58870)  
3. JUCE Best Practices: Claude Code Skill for Audio Dev \- MCP Market, accessed April 30, 2026, [https://mcpmarket.com/tools/skills/juce-best-practices](https://mcpmarket.com/tools/skills/juce-best-practices)  
4. Does ORT have any real-time safety guarantees, i.e. that it's non-blocking? · Issue \#15303 · microsoft/onnxruntime \- GitHub, accessed April 30, 2026, [https://github.com/microsoft/onnxruntime/issues/15303](https://github.com/microsoft/onnxruntime/issues/15303)  
5. Best coding practices for audio applications \- 2 questions (both answered) \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/best-coding-practices-for-audio-applications-2-questions-both-answered/32297](https://forum.juce.com/t/best-coding-practices-for-audio-applications-2-questions-both-answered/32297)  
6. ML into VST plugin : r/JUCE \- Reddit, accessed April 30, 2026, [https://www.reddit.com/r/JUCE/comments/1eqa05k/ml\_into\_vst\_plugin/](https://www.reddit.com/r/JUCE/comments/1eqa05k/ml_into_vst_plugin/)  
7. ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications | Request PDF \- ResearchGate, accessed April 30, 2026, [https://www.researchgate.net/publication/384712488\_ANIRA\_An\_Architecture\_for\_Neural\_Network\_Inference\_in\_Real-Time\_Audio\_Applications](https://www.researchgate.net/publication/384712488_ANIRA_An_Architecture_for_Neural_Network_Inference_in_Real-Time_Audio_Applications)  
8. AI audio plugin Idea, how to deal with buffer size? \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/ai-audio-plugin-idea-how-to-deal-with-buffer-size/65144](https://forum.juce.com/t/ai-audio-plugin-idea-how-to-deal-with-buffer-size/65144)  
9. Reading/writing values lock free to/from processBlock \- Getting Started \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947](https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947)  
10. Realtime Safe ValueTree (Non-Locking Solution) \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/realtime-safe-valuetree-non-locking-solution/65955](https://forum.juce.com/t/realtime-safe-valuetree-non-locking-solution/65955)  
11. Standard approach to maintaining shared State with Processor and Editor(s) \- Audio Plugins, accessed April 30, 2026, [https://forum.juce.com/t/standard-approach-to-maintaining-shared-state-with-processor-and-editor-s/43621](https://forum.juce.com/t/standard-approach-to-maintaining-shared-state-with-processor-and-editor-s/43621)  
12. Lock-free queues and visualization of data \- Audio Plugins \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/lock-free-queues-and-visualization-of-data/20659](https://forum.juce.com/t/lock-free-queues-and-visualization-of-data/20659)  
13. llama.cpp is all you need : r/LocalLLaMA \- Reddit, accessed April 30, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1j417qh/llamacpp\_is\_all\_you\_need/](https://www.reddit.com/r/LocalLLaMA/comments/1j417qh/llamacpp_is_all_you_need/)  
14. ggml-org/llama.cpp: LLM inference in C/C++ \- GitHub, accessed April 30, 2026, [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
15. library \- Ollama, accessed April 30, 2026, [https://ollama.com/library](https://ollama.com/library)  
16. DirtyBeastAfterTheToad/LLMidi: MIDI generator plugin (vst3) powered by local or online large language models (LLMs). \- GitHub, accessed April 30, 2026, [https://github.com/DirtyBeastAfterTheToad/LLMidi](https://github.com/DirtyBeastAfterTheToad/LLMidi)  
17. Llama.cpp wrapper \- Plugins \- Xojo Programming Forum, accessed April 30, 2026, [https://forum.xojo.com/t/llama-cpp-wrapper/86682](https://forum.xojo.com/t/llama-cpp-wrapper/86682)  
18. getnamo/Llama-Unreal: Llama.cpp plugin for Unreal Engine 5 \- GitHub, accessed April 30, 2026, [https://github.com/getnamo/Llama-Unreal](https://github.com/getnamo/Llama-Unreal)  
19. Accelerating Phi-2, CodeLlama, Gemma and other Gen AI models with ONNX Runtime, accessed April 30, 2026, [https://onnxruntime.ai/blogs/accelerating-phi-2](https://onnxruntime.ai/blogs/accelerating-phi-2)  
20. Introduction \- Ollama's documentation, accessed April 30, 2026, [https://docs.ollama.com/api/introduction](https://docs.ollama.com/api/introduction)  
21. GitHub \- jmont-dev/ollama-hpp: Modern, Header-only C++ bindings for the Ollama API., accessed April 30, 2026, [https://github.com/jmont-dev/ollama-hpp](https://github.com/jmont-dev/ollama-hpp)  
22. Libraries & SDKs \- Ollama \- Mintlify, accessed April 30, 2026, [https://www.mintlify.com/ollama/ollama/integrations/libraries](https://www.mintlify.com/ollama/ollama/integrations/libraries)  
23. Question on calling a REST API using juce::Thread, accessed April 30, 2026, [https://forum.juce.com/t/question-on-calling-a-rest-api-using-juce-thread/52370](https://forum.juce.com/t/question-on-calling-a-rest-api-using-juce-thread/52370)  
24. Inference \- ONNX Runtime, accessed April 30, 2026, [https://onnxruntime.ai/inference](https://onnxruntime.ai/inference)  
25. ONNX Runtime Execution Providers, accessed April 30, 2026, [https://onnxruntime.ai/docs/execution-providers/](https://onnxruntime.ai/docs/execution-providers/)  
26. Deploying an ONNX model \- Audio Plugins \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/deploying-an-onnx-model/59753](https://forum.juce.com/t/deploying-an-onnx-model/59753)  
27. ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications \- arXiv, accessed April 30, 2026, [https://arxiv.org/pdf/2506.12665](https://arxiv.org/pdf/2506.12665)  
28. Thread management | onnxruntime, accessed April 30, 2026, [https://onnxruntime.ai/docs/performance/tune-performance/threading.html](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)  
29. Right way in include dylib/DLLs for installer \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/right-way-in-include-dylib-dlls-for-installer/60289](https://forum.juce.com/t/right-way-in-include-dylib-dlls-for-installer/60289)  
30. Neural Audio Processing on Android Phones, accessed April 30, 2026, [https://www.dafx.de/paper-archive/2024/papers/DAFx24\_paper\_78.pdf](https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_78.pdf)  
31. olilarkin/awesome-musicdsp: A curated list of my favourite music DSP and audio programming resources \- GitHub, accessed April 30, 2026, [https://github.com/olilarkin/awesome-musicdsp](https://github.com/olilarkin/awesome-musicdsp)  
32. Real-Time Inference of Neural Networks \- The Audio Developer Conference, accessed April 30, 2026, [https://data.audio.dev/talks/2024/real-time-inference-of-neural-networks/slides.pdf](https://data.audio.dev/talks/2024/real-time-inference-of-neural-networks/slides.pdf)  
33. A Comparison of Deep Learning Inference Engines for Embedded Real-Time Audio Classification \- ResearchGate, accessed April 30, 2026, [https://www.researchgate.net/publication/363511029\_A\_Comparison\_of\_Deep\_Learning\_Inference\_Engines\_for\_Embedded\_Real-Time\_Audio\_Classification](https://www.researchgate.net/publication/363511029_A_Comparison_of_Deep_Learning_Inference_Engines_for_Embedded_Real-Time_Audio_Classification)  
34. Homepage — Essentia 2.1-beta6-dev documentation, accessed April 30, 2026, [https://essentia.upf.edu/](https://essentia.upf.edu/)  
35. Essentia: An Open-source Library for Audio Analysis | Papers We Love, accessed April 30, 2026, [https://paperswelove.org/papers/essentia-an-open-source-library-for-audio-analysis-c23965b8/](https://paperswelove.org/papers/essentia-an-open-source-library-for-audio-analysis-c23965b8/)  
36. Music extractor — Essentia 2.1-beta6-dev documentation, accessed April 30, 2026, [https://essentia.upf.edu/streaming\_extractor\_music.html](https://essentia.upf.edu/streaming_extractor_music.html)  
37. Foundation Models for Music: A Survey \- arXiv, accessed April 30, 2026, [https://arxiv.org/html/2408.14340v3](https://arxiv.org/html/2408.14340v3)  
38. Audio signal feature extraction for analysis | by Athina B \- Medium, accessed April 30, 2026, [https://athina-b.medium.com/audio-signal-feature-extraction-for-analysis-507861717dc1](https://athina-b.medium.com/audio-signal-feature-extraction-for-analysis-507861717dc1)  
39. Audio ML Pipelines in C | PDF | Spectral Density | Digital Signal Processing \- Scribd, accessed April 30, 2026, [https://www.scribd.com/document/893008051/Project](https://www.scribd.com/document/893008051/Project)  
40. essentia/FAQ.md at master \- GitHub, accessed April 30, 2026, [https://github.com/MTG/essentia/blob/master/FAQ.md](https://github.com/MTG/essentia/blob/master/FAQ.md)  
41. Essentia documentation contents, accessed April 30, 2026, [https://essentia.upf.edu/documentation/contents.html](https://essentia.upf.edu/documentation/contents.html)  
42. An Evaluation of Audio Feature Extraction Toolboxes \- NTNU, accessed April 30, 2026, [https://www.ntnu.edu/documents/1001201110/1266017954/DAFx-15\_submission\_43\_v2.pdf](https://www.ntnu.edu/documents/1001201110/1266017954/DAFx-15_submission_43_v2.pdf)  
43. Audio tagging — sherpa 1.3 documentation, accessed April 30, 2026, [https://k2-fsa.github.io/sherpa/onnx/audio-tagging/index.html](https://k2-fsa.github.io/sherpa/onnx/audio-tagging/index.html)  
44. Sherpa-ONNX Hotwords Biasing Guide | PDF | We Chat | Graphics Processing Unit \- Scribd, accessed April 30, 2026, [https://www.scribd.com/document/795289962/Sherpa](https://www.scribd.com/document/795289962/Sherpa)  
45. Pre-trained models — sherpa 1.3 documentation, accessed April 30, 2026, [https://k2-fsa.github.io/sherpa/onnx/audio-tagging/pretrained\_models.html](https://k2-fsa.github.io/sherpa/onnx/audio-tagging/pretrained_models.html)  
46. SLAM-AAC: Enhancing Audio Captioning with Paraphrasing Augmentation and CLAP-Refine through LLMs \- arXiv, accessed April 30, 2026, [https://arxiv.org/html/2410.09503v1](https://arxiv.org/html/2410.09503v1)  
47. CLAP \- Hugging Face, accessed April 30, 2026, [https://huggingface.co/docs/transformers/model\_doc/clap](https://huggingface.co/docs/transformers/model_doc/clap)  
48. LAION-AI/CLAP: Contrastive Language-Audio Pretraining \- GitHub, accessed April 30, 2026, [https://github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)  
49. SynthScribe: Deep Multimodal Tools for Synthesizer Sound Retrieval and Exploration \- arXiv, accessed April 30, 2026, [https://arxiv.org/html/2312.04690v2](https://arxiv.org/html/2312.04690v2)  
50. SynthScribe: Deep Multimodal Tools for Synthesizer Sound Retrieval and Exploration \- GitHub Pages, accessed April 30, 2026, [https://neuripscreativityworkshop.github.io/2023/papers/ml4cd2023\_paper31.pdf](https://neuripscreativityworkshop.github.io/2023/papers/ml4cd2023_paper31.pdf)  
51. TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control \- arXiv, accessed April 30, 2026, [https://arxiv.org/html/2603.09332v1](https://arxiv.org/html/2603.09332v1)  
52. TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control \- arXiv, accessed April 30, 2026, [https://arxiv.org/pdf/2603.09332](https://arxiv.org/pdf/2603.09332)  
53. Qwen3-Omni Technical Report \- arXiv, accessed April 30, 2026, [https://arxiv.org/html/2509.17765v1](https://arxiv.org/html/2509.17765v1)  
54. The official repo of Qwen2-Audio chat & pretrained large audio language model proposed by Alibaba Cloud. \- GitHub, accessed April 30, 2026, [https://github.com/qwenlm/qwen2-audio](https://github.com/qwenlm/qwen2-audio)  
55. Qwen/Qwen3-Omni-30B-A3B-Captioner \- Hugging Face, accessed April 30, 2026, [https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)  
56. Structured Prompting with JSON: The Engineering Path to Reliable LLMs | by vishal dutt, accessed April 30, 2026, [https://medium.com/@vishal.dutt.data.architect/structured-prompting-with-json-the-engineering-path-to-reliable-llms-2c0cb1b767cf](https://medium.com/@vishal.dutt.data.architect/structured-prompting-with-json-the-engineering-path-to-reliable-llms-2c0cb1b767cf)  
57. JSON prompting for LLMs \- IBM Developer, accessed April 30, 2026, [https://developer.ibm.com/articles/json-prompting-llms/](https://developer.ibm.com/articles/json-prompting-llms/)  
58. Prompting-Based Synthetic Data Generation \- Emergent Mind, accessed April 30, 2026, [https://www.emergentmind.com/topics/prompting-based-synthetic-data-generation](https://www.emergentmind.com/topics/prompting-based-synthetic-data-generation)  
59. Prompting Techniques | Prompt Engineering Guide, accessed April 30, 2026, [https://www.promptingguide.ai/techniques](https://www.promptingguide.ai/techniques)  
60. LLM Engineering (Part I) \- Medium, accessed April 30, 2026, [https://medium.com/@yugalnandurkar5/llm-engineering-part-i-fa48d4307d26](https://medium.com/@yugalnandurkar5/llm-engineering-part-i-fa48d4307d26)  
61. AI-assisted JSON Schema Creation and Mapping Deutsche Forschungsgemeinschaft (DFG) under project numbers 528693298 (preECO), 358283783 (SFB1333), and 390740016 (EXC2075) \- arXiv, accessed April 30, 2026, [https://arxiv.org/html/2508.05192v2](https://arxiv.org/html/2508.05192v2)  
62. MUSIC FLAMINGO: SCALING MUSIC UNDERSTANDING IN AUDIO LANGUAGE MODELS \- OpenReview, accessed April 30, 2026, [https://openreview.net/pdf?id=RS7T9S16Bl](https://openreview.net/pdf?id=RS7T9S16Bl)  
63. grammars/README.md · 6a2f0b3474d479bda4ac2ee7cfd5dcdcf0be1f79 · aigc/llama.cpp \- https://cnb.cool, accessed April 30, 2026, [https://cnb.cool/aigc/llama.cpp/-/blob/6a2f0b3474d479bda4ac2ee7cfd5dcdcf0be1f79/grammars/README.md](https://cnb.cool/aigc/llama.cpp/-/blob/6a2f0b3474d479bda4ac2ee7cfd5dcdcf0be1f79/grammars/README.md)  
64. llama.cpp \- README.md \- GitLab, accessed April 30, 2026, [https://gitlab.informatik.uni-halle.de/ambcj/llama.cpp/-/blob/0c4d489e29e53589bf13a801fe7c94b7b546d8f6/README.md](https://gitlab.informatik.uni-halle.de/ambcj/llama.cpp/-/blob/0c4d489e29e53589bf13a801fe7c94b7b546d8f6/README.md)  
65. llama.cpp/grammars/README.md at master · ggml-org/llama.cpp · GitHub, accessed April 30, 2026, [https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)  
66. GBNF Function Calling Grammar Generator for llama.cpp to make function calling with every model supporting grammar based sampling. (most models, I only had problems with Deepseek Code Instruct) \- Reddit, accessed April 30, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/187rjz5/gbnf\_function\_calling\_grammar\_generator\_for/](https://www.reddit.com/r/LocalLLaMA/comments/187rjz5/gbnf_function_calling_grammar_generator_for/)  
67. Tutorial: Saving and loading your plug-in state \- JUCE, accessed April 30, 2026, [https://juce.com/tutorials/tutorial\_audio\_processor\_value\_tree\_state/](https://juce.com/tutorials/tutorial_audio_processor_value_tree_state/)  
68. JUCE/modules/juce\_audio\_processors/utilities/juce\_AudioProcessorValueTreeState.cpp at master \- GitHub, accessed April 30, 2026, [https://github.com/juce-framework/JUCE/blob/master/modules/juce\_audio\_processors/utilities/juce\_AudioProcessorValueTreeState.cpp](https://github.com/juce-framework/JUCE/blob/master/modules/juce_audio_processors/utilities/juce_AudioProcessorValueTreeState.cpp)  
69. AudioProcessorValueTreeState Parameter Access Best Practices \- Audio Plugins \- JUCE, accessed April 30, 2026, [https://forum.juce.com/t/audioprocessorvaluetreestate-parameter-access-best-practices/38981](https://forum.juce.com/t/audioprocessorvaluetreestate-parameter-access-best-practices/38981)  
70. Understanding Lock in Audio Thread \- Audio Plugins \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/understanding-lock-in-audio-thread/60007](https://forum.juce.com/t/understanding-lock-in-audio-thread/60007)  
71. Update text on changed value in audio processor \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/update-text-on-changed-value-in-audio-processor/44169](https://forum.juce.com/t/update-text-on-changed-value-in-audio-processor/44169)  
72. Audio parameters not preserved when plugin is updated \- JUCE Forum, accessed April 30, 2026, [https://forum.juce.com/t/audio-parameters-not-preserved-when-plugin-is-updated/62954](https://forum.juce.com/t/audio-parameters-not-preserved-when-plugin-is-updated/62954)  
73. Tutorial: Wavetable synthesis \- JUCE, accessed April 30, 2026, [https://juce.com/tutorials/tutorial\_wavetable\_synth/](https://juce.com/tutorials/tutorial_wavetable_synth/)  
74. Physical Modeling Explored \- Sampling vs Modeling \- YouTube, accessed April 30, 2026, [https://www.youtube.com/watch?v=RHT0SNu0Cuk](https://www.youtube.com/watch?v=RHT0SNu0Cuk)  
75. awesome-audio-dsp/sections/CODE\_LIBRARIES.md at main \- GitHub, accessed April 30, 2026, [https://github.com/BillyDM/awesome-audio-dsp/blob/main/sections/CODE\_LIBRARIES.md](https://github.com/BillyDM/awesome-audio-dsp/blob/main/sections/CODE_LIBRARIES.md)  
76. C++ LIBRARIES FOR BASIC DSP FUNCTIONS \- Reddit, accessed April 30, 2026, [https://www.reddit.com/r/DSP/comments/1k3k47q/c\_libraries\_for\_basic\_dsp\_functions/](https://www.reddit.com/r/DSP/comments/1k3k47q/c_libraries_for_basic_dsp_functions/)  
77. Maximilian: C++ Audio and Music DSP Library \- Mick Grierson \- JUCE Summit 2015, accessed April 30, 2026, [https://www.youtube.com/watch?v=H-Av78mtFF4](https://www.youtube.com/watch?v=H-Av78mtFF4)  
78. \[SOLUTION\] to using Maximilian on Juce 6 \+ Visual Studio 2019 \- The Audio Programmer Juce Tutorial 21 \- Getting Started, accessed April 30, 2026, [https://forum.juce.com/t/solution-to-using-maximilian-on-juce-6-visual-studio-2019-the-audio-programmer-juce-tutorial-21/41200](https://forum.juce.com/t/solution-to-using-maximilian-on-juce-6-visual-studio-2019-the-audio-programmer-juce-tutorial-21/41200)  
79. DaisySP away from the hardware \- (AKA Plugins with JUCE\!) \- Daisy Forums, accessed April 30, 2026, [https://forum.electro-smith.com/t/daisysp-away-from-the-hardware-aka-plugins-with-juce/1106](https://forum.electro-smith.com/t/daisysp-away-from-the-hardware-aka-plugins-with-juce/1106)  
80. DaisySP \- GitHub Pages, accessed April 30, 2026, [https://electro-smith.github.io/DaisySP/](https://electro-smith.github.io/DaisySP/)  
81. Optimizing performance \- Software Development \- Daisy Forums, accessed April 30, 2026, [https://forum.electro-smith.com/t/optimizing-performance/5366](https://forum.electro-smith.com/t/optimizing-performance/5366)  
82. Developing Procedural Generation Tools for Video Game Audio Designers, accessed April 30, 2026, [https://openaccess.wgtn.ac.nz/articles/thesis/Developing\_Procedural\_Generation\_Tools\_for\_Video\_Game\_Audio\_Designers/17134286/2/files/31684355.pdf](https://openaccess.wgtn.ac.nz/articles/thesis/Developing_Procedural_Generation_Tools_for_Video_Game_Audio_Designers/17134286/2/files/31684355.pdf)  
83. Frequency domain simulation of temporal domain processes, FFT stuff \- DSP and Plugin Development Forum \- KVR Audio, accessed April 30, 2026, [https://www.kvraudio.com/forum/viewtopic.php?p=8891931](https://www.kvraudio.com/forum/viewtopic.php?p=8891931)  
84. News | discoDSP, accessed April 30, 2026, [https://www.discodsp.com/news/archive/](https://www.discodsp.com/news/archive/)  
85. New modern C++ library for fast decramped modulable filters \- DSP and Plugin Development Forum \- KVR Audio, accessed April 30, 2026, [https://www.kvraudio.com/forum/viewtopic.php?t=625630](https://www.kvraudio.com/forum/viewtopic.php?t=625630)  
86. Tutorial: Physical Modeling Synthesis for Max Users: A Primer | Cycling '74, accessed April 30, 2026, [https://cycling74.com/2012/10/09/physical-modeling-synthesis-for-max-users-a-primer](https://cycling74.com/2012/10/09/physical-modeling-synthesis-for-max-users-a-primer)