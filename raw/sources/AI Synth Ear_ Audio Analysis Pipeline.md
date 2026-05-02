# **Architectural Design and Implementation of the "Ear" Module: Audio Feature Extraction and Semantic Analysis in C++**

The evolution of generative artificial intelligence has catalyzed a paradigm shift in digital audio workstation (DAW) environments. Integrating Large Language Models (LLMs) into audio synthesis workflows requires establishing a deterministic, low-latency bridge between the high-frequency domain of digital signal processing (DSP) and the discrete, semantic domain of natural language processing. In the architectural design of "Sound Forge," a hybrid VST3 synthesizer plugin engineered in modern C++ utilizing the JUCE framework, this critical interface is designated as the "Ear" module.

The primary directive of the Ear module is to continuously monitor the synthesizer's real-time audio output, extract a multidimensional array of acoustic and neural features, and synthesize these characteristics into a coherent semantic prompt. This prompt informs an autonomous LLM agent (the "Brain"), granting it the acoustic awareness necessary to execute parametric modifications aligned with user intent. The execution of this module is subject to extreme operational constraints: it must operate within the strict boundaries of non-preemptible audio threads, perform complex mathematical and neural operations without introducing latency, and manage memory safely to prevent priority inversions or buffer underruns.

This research report presents an exhaustive, highly technical analysis of the architectures, algorithms, and deployment paradigms necessary to construct the Ear module. The investigation rigorously details algorithmic feature extraction utilizing the Essentia C++ library, real-time neural audio tagging via Zipformer architectures deployed on ONNX Runtime, texture-aware multimodal semantic embeddings (CLAP and TRR), and the implementation of a C++ data pipeline guaranteeing absolute lock-free thread safety.

## **1\. Algorithmic DSP Feature Extraction: The CPU-Safe Route**

While deep neural networks provide sophisticated high-level semantic labeling, traditional algorithmic feature extraction remains an indispensable component of the Ear module. Algorithmic DSP operates with deterministic execution times, requires significantly fewer CPU cycles than neural inference, and provides mathematically objective metrics (such as precise spectral distribution and transient locations) that neural networks often obfuscate. To supply the LLM with a highly accurate acoustic baseline, the system must extract features representing the instantaneous timbral shift of the synthesizer.

### **1.1 Comparative Analysis of C++ Audio Analysis Libraries**

The integration of a robust Music Information Retrieval (MIR) library is foundational to the DSP extraction layer. Within the C++ ecosystem, several state-of-the-art libraries exist, including Aubio, Marsyas, LibXtract, Yaafe, and Essentia.1

Extensive benchmarking has been conducted to evaluate the computational efficiency of these toolboxes. In comparative studies processing over 16.5 hours of diverse audio data to calculate Mel-Frequency Cepstral Coefficients (MFCCs)—a standard metric for timbral representation—native C/C++ libraries consistently outperformed interpreted or wrapped languages.1 For example, pure-Python implementations like Librosa required 1 hour and 53 minutes to complete the dataset analysis, rendering them fundamentally unsuitable for real-time VST3 integration.1 Conversely, Yaafe executed the task in 3 minutes and 30 seconds, while Essentia completed the analysis in 4 minutes and 12 seconds.1 Aubio, while highly efficient for fundamental frequency tracking and high-level segmentation, lacks the exhaustive descriptive taxonomy provided by Essentia.1

Given the necessity for a broad spectrum of spectral, temporal, and tonal descriptors, Essentia emerges as the optimal library for the Ear module. Released under the Affero GPLv3 license, Essentia contains a vast collection of optimized, reusable algorithms that implement standard digital signal processing blocks and statistical characterizations specifically designed for MIR.3

### **1.2 Essentia Streaming Architecture for Real-Time Execution**

Essentia's architecture is bifurcated into two primary operational modes: Standard and Streaming.6 The Standard mode relies on push-based, block-by-block computation, which is highly effective for offline analysis but introduces significant overhead when managing continuous buffers. The Streaming mode, however, models computation as a Directed Acyclic Graph (DAG), pulling data through a network of connected algorithms via Sources and Sinks.8 For a real-time VST3 plugin, the Streaming architecture is paramount.

Integrating a continuous audio stream into an Essentia DAG presents specific architectural challenges. Historical approaches attempted to utilize VectorInput sinks to inject arrays of floats; however, real-time developers have documented severe synchronization bottlenecks and exceptions (e.g., VectorInput is not connected to anything...) when applying this to continuous callbacks.6

The robust, industry-standard solution for real-time audio injection into an Essentia network relies on the RingBufferInput algorithm.12 This input node is explicitly engineered to bridge asynchronous host environments with the internal processing loop of the DAG. The topological layout of the DSP pipeline is constructed as follows:

1. **Ingestion Node (RingBufferInput):** Operates as the origin point of the DAG, accepting floating-point audio data pushed from the plugin's background thread.12  
2. **Segmentation Node (FrameCutter):** Receives the continuous stream and segments it into overlapping windows. For robust timbral analysis, a frame size of 1024 or 2048 samples with a 50% hop size (512 or 1024 samples) is typically configured.2  
3. **Windowing Node (Windowing):** Applies a mathematical windowing function (such as Hann, Hamming, or Blackman-Harris) to the time-domain frame to mitigate spectral leakage at the boundaries prior to frequency domain transformation.2  
4. **Transformation Node (Spectrum):** Executes a Fast Fourier Transform (FFT) on the windowed frame, outputting the magnitude spectrum required for downstream spectral descriptors.2  
5. **Descriptor Nodes (Parallel Execution):**  
   * **MFCC:** Attaches to the Spectrum output. It warps the linear frequency spectrum into the perceptually scaled Mel scale (e.g., using 128 bands) and applies a Discrete Cosine Transform (DCT) to yield cepstral coefficients, which compactly describe the spectral envelope.2  
   * **Centroid:** Calculates the spectral center of mass, which correlates strongly with the human perception of "brightness".9  
   * **ZeroCrossingRate (ZCR):** Bypasses the FFT, attaching directly to the FrameCutter output. It quantifies the rate at which the signal changes sign, serving as a highly effective metric for distinguishing harmonic content from broadband noise (e.g., separating a sine wave sub-bass from white noise modulation).15  
   * **OnsetDetection:** Evaluates spectral flux or complex domain deviations across sequential frames to identify transient peaks, alerting the LLM to rhythmic events or sharp envelope attacks.13  
6. **Aggregation Node (PoolStorage or RingBufferOutput):** Acts as the terminal sink, accumulating the computed scalar values and vectors into an Essentia Pool for subsequent JSON serialization and delivery.13

### **1.3 Thread Safety Protocols and JUCE Integration Mechanics**

The foundational axiom of audio plugin development is the strict adherence to real-time thread constraints. The host DAW invokes the plugin's processBlock function on a high-priority, non-preemptible thread. This function must execute deterministically within a strict deadline (often less than 2 milliseconds for a buffer size of 64 samples at 44.1kHz).18 If the thread blocks, it causes audio dropouts, glitches, and potential host crashes.18

Consequently, the processBlock function is prohibited from utilizing standard threading primitives such as std::mutex, std::unique\_lock, or spinlocks. It must not perform memory allocations (new, malloc, or resizing std::vector), execute disk I/O, or perform unconstrained iterative operations.19 Executing heavy Essentia algorithms or ONNX neural network inferences directly within processBlock violates these constraints entirely and guarantees systemic failure.18

Furthermore, if a low-priority thread attempts to compute DSP features while holding a lock that the high-priority audio thread subsequently requests, a priority inversion occurs. The audio thread is forced to sleep, missing its deadline while waiting for the OS scheduler to resolve the lock contention.21

#### **Lock-Free Single-Producer, Single-Consumer (SPSC) Architecture**

To resolve this, the audio thread must function exclusively as a data producer, offloading all analysis to a secondary, lower-priority background thread (inheriting from juce::Thread).21 Data is securely handed off via a lock-free Single-Producer, Single-Consumer (SPSC) circular queue. In the JUCE framework, this is facilitated by juce::AbstractFifo operating in conjunction with a pre-allocated memory buffer (such as std::vector\<float\> or juce::AudioBuffer\<float\>).18

The AbstractFifo class does not hold the data; it manages the read and write atomic indices of a theoretical circular buffer. Because the buffer is circular, writing a contiguous block of incoming audio samples may require splitting the data if the write pointer approaches the end of the allocated memory array and must wrap around to index 0\.23

**Producer Implementation (Audio Thread):** The audio thread polls the FIFO for available capacity. If space permits, it calls prepareToWrite, which populates four integer variables representing the start indices and block sizes for the potentially split data.23 The audio data is then copied into the shared buffer, and the write indices are atomically advanced.

C++

// Inside processBlock()  
void EarModule::pushAudioToFifo(const juce::AudioBuffer\<float\>& buffer) {  
    const int numSamples \= buffer.getNumSamples();  
      
    // Check if the FIFO has enough free space to avoid buffer overruns  
    if (abstractFifo.getFreeSpace() \< numSamples) return;   
      
    int start1, block1, start2, block2;  
    abstractFifo.prepareToWrite(numSamples, start1, block1, start2, block2);  
      
    // Highly optimized SIMD copy operations  
    if (block1 \> 0) {  
        juce::FloatVectorOperations::copy(sharedBuffer.data() \+ start1,   
                                          buffer.getReadPointer(0), block1);  
    }  
    if (block2 \> 0) {  
        juce::FloatVectorOperations::copy(sharedBuffer.data() \+ start2,   
                                          buffer.getReadPointer(0) \+ block1, block2);  
    }  
      
    // Atomically commit the write, making the data visible to the consumer  
    abstractFifo.finishedWrite(block1 \+ block2);  
      
    // Signal the background thread using a lock-free mechanism  
    backgroundThread.notify();   
}

**Consumer Implementation (Background Thread):** The background thread, operating in its continuous run() loop, sleeps until notified or until a specific timeout occurs. Upon waking, it reads the data using prepareToRead, feeding the floating-point values into the Essentia RingBufferInput.12

C++

// Inside background juce::Thread::run()  
void BackgroundAnalyzer::run() {  
    while (\!threadShouldExit()) {  
        if (event.wait(10)) { // Wait for notification from processBlock  
            int numReady \= abstractFifo.getNumReady();  
            if (numReady \> 0) {  
                int start1, block1, start2, block2;  
                abstractFifo.prepareToRead(numReady, start1, block1, start2, block2);  
                  
                std::vector\<float\> readBuffer(numReady);  
                  
                if (block1 \> 0) {  
                    std::copy(sharedBuffer.begin() \+ start1,   
                              sharedBuffer.begin() \+ start1 \+ block1,   
                              readBuffer.begin());  
                }  
                if (block2 \> 0) {  
                    std::copy(sharedBuffer.begin() \+ start2,   
                              sharedBuffer.begin() \+ start2 \+ block2,   
                              readBuffer.begin() \+ block1);  
                }  
                  
                abstractFifo.finishedRead(block1 \+ block2);  
                  
                // Route the data to Essentia and Neural pipelines  
                dispatchToEssentia(readBuffer);  
                dispatchToNeuralNetwork(readBuffer);  
            }  
        }  
    }  
}

By ensuring that the SPSC queue utilizes explicit memory barriers (which are handled internally by juce::AbstractFifo via std::atomic operations), the system guarantees that cache lines are synchronized across multi-core processors without requiring context-switching locks, preserving the integrity of the real-time audio pipeline.

## **2\. Neural Audio Tagging & Semantic Classification**

While the deterministic DSP features (such as MFCCs and ZCR) describe the mathematical realities of the acoustic waveform, a profound semantic gap exists between these numbers and the perceptual ontology of an LLM. An LLM operates almost exclusively in the domain of human language; it intuitively understands complex adjectives like "distorted," "plucky," "warm," or "sub-bass" significantly better than it interprets "a high zero-crossing rate combined with spectral energy concentrated below 60Hz." Bridging this semantic gap necessitates the implementation of Neural Audio Tagging—the process of passing the collected audio buffer through a pre-trained neural network to emit text-based labels.

### **2.1 Lightweight Classifiers: The sherpa-onnx Framework and Zipformers**

The integration of deep learning models directly into a compiled VST3 plugin introduces severe constraints regarding binary payload size, volatile memory (RAM) usage, and inference latency. State-of-the-art audio event detection mechanisms typically utilize heavyweight Audio Spectrogram Transformers (AST), which demand substantial computational resources and gigabytes of memory. However, the Ear module requires extreme efficiency.

The sherpa-onnx framework presents a highly optimized alternative, specifically engineered for the edge deployment of speech and audio classification models.26 sherpa-onnx provides a streamlined C/C++ API that wraps the ONNX Runtime execution engine, eliminating the need for bulky Python dependencies or PyTorch binaries in the final plugin package.26

For the specific task of generating semantic acoustic labels, sherpa-onnx supports AudioSet-trained Zipformer models, notably the sherpa-onnx-zipformer-small-audio-tagging-2024-04-15 checkpoint.26 Zipformers represent an advanced evolution of the Transformer architecture specifically tailored for acoustic and sequential data processing. Unlike traditional Transformers that maintain a constant sequence length across all layers, Zipformers utilize an hourglass-like downsampling and upsampling scheme, combined with block-local self-attention. This architectural design drastically reduces the temporal dimension of the sequence at the deeper layers of the network, resulting in an exponential reduction in computational complexity (FLOPs) and inference latency.31

To further mitigate resource consumption, the targeted Zipformer model is provided as an INT8 quantized ONNX graph.31 Quantization reduces the precision of the network's weights from 32-bit floating-point numbers to 8-bit integers, effectively compressing the model.int8.onnx file to a mere 26 megabytes, making it highly suitable for seamless inclusion within a plugin bundle.31

Despite its diminutive size, the model is pre-trained to recognize an expansive class vocabulary detailed in a corresponding class\_labels\_indices.csv file. The network's vocabulary encompasses a wide range of acoustic phenomena, instrumental concepts, and environmental sounds, detecting labels such as "Cat," "Music," "Laughter," and musical artifacts like distortion or specific tonal registers.26

### **2.2 C++ Classification Layer Implementation via ONNX Runtime**

To construct a high-velocity classification layer, the background thread must accumulate a continuous 1-second acoustic buffer (e.g., 44,100 samples assuming a 44.1kHz sample rate) generated from the lock-free FIFO mechanism. This 1-second window provides sufficient temporal context for the neural network to identify sustained tones, rhythms, and textural artifacts.

The C++ integration of the Zipformer model leverages the ONNX Runtime (Ort::Session) infrastructure, abstracted partially by sherpa-onnx or implemented directly via the raw C++ APIs.29 The initialization sequence, which inherently requires dynamic memory allocation and file system I/O, must be executed exclusively during the plugin's prepareToPlay callback or within the constructor of the module, strictly bypassing the audio thread.

**Initialization Phase:** The Ort::Env and Ort::SessionOptions are configured. To maintain deterministic cross-platform compatibility across various consumer machines running macOS and Windows, relying on hardware-specific execution providers (such as CUDA or CoreML) introduces significant instability.32 Therefore, the system utilizes the highly optimized CPUExecutionProvider. Performance is maximized by enabling OpenMP threading options and instructing the compiler to leverage advanced vector extensions like AVX/AVX2.29

**Inference Execution (Background Thread):**

1. **Feature Extraction:** The raw 1-second audio buffer cannot be directly ingested by the Zipformer. It must first be converted into acoustic features. The model expects an 80-dimensional or 128-dimensional Mel-filterbank sequence, generated using standard DSP parameters (e.g., a 25ms window length and a 10ms frame shift).35  
2. **Tensor Instantiation:** The computed Mel features are sequentially aligned in memory and wrapped into an Ort::Value tensor. Because the ONNX Runtime requires strict, contiguous memory alignments, Ort::MemoryInfo::CreateCpu is invoked to map the std::vector\<float\> safely to the tensor dimensions.29  
3. **Forward Pass Computation:** The computational graph is executed via session.Run(). This operation is CPU-bound and blocking, emphasizing why it must be relegated to the background thread.29  
4. **Softmax Normalization and Top-K Selection:** The terminal layer of the Zipformer emits a vector of unnormalized raw logits corresponding to each label in the vocabulary. A softmax activation function is applied to normalize these values into a probability distribution ranging from 0.0 to 1.0. A sorting algorithm isolates the top-K highest probabilities.31

The resulting output is structurally formatted as a collection of AudioEvent objects containing a string representation of the semantic label and a numerical confidence score (e.g., AudioEvent(name="Distorted Guitar", index=42, prob=0.947)).31

| Architectural Phase | Assigned Thread Context | Operational Constraints | Target Mechanism |
| :---- | :---- | :---- | :---- |
| **Model Instantiation** | Main / GUI Thread | High latency tolerance, dynamic memory allocation permissible. | Ort::Session initialization, loading .onnx binaries and parsing .csv label indices. |
| **Buffer Aggregation** | Background Thread | Must implement lock-free reads from the SPSC FIFO. | AbstractFifo read block translation into std::vector\<float\>. |
| **Neural Inference Pass** | Background Thread | Heavy CPU cycle utilization, strictly blocking execution. | Ort::Session::Run(), output tensor parsing, softmax, and Top-K extraction. |

*Table 1: Thread boundary assignments and operational constraints for the Neural Audio Tagging sub-system.*

## **3\. Multimodal Semantic Embeddings: CLAP & Texture Resonance Retrieval (TRR)**

While discrete neural tagging provides explicit semantic categorization, it exhibits a fundamental limitation: rigidity. Audio tagging models are strictly constrained to outputting labels upon which they were explicitly trained. If the synthesizer generates an entirely novel acoustic texture—such as a complex frequency-modulated granular drone—a rigid classifier will fail to identify it, returning either low-confidence unrelated tags or no tags at all.

To grant the LLM genuine, open-ended insight into the synthesizer's output, the Ear module must implement Multimodal Semantic Embeddings. This approach maps audio waveforms into a continuous, high-dimensional mathematical vector space shared concurrently by textual descriptions, facilitating open-vocabulary comparisons and zero-shot reasoning.

### **3.1 Contrastive Language-Audio Pretraining (CLAP) Integration**

CLAP, spearheaded by organizations such as LAION, adapts the highly successful contrastive learning architecture of OpenAI's CLIP, applying it to audio-text pairs.39 Through exhaustive pretraining on massive datasets comprising audio clips and corresponding textual captions, a CLAP model forces the mathematical embedding of an audio clip (e.g., the waveform of a sawtooth bass) to achieve maximum cosine similarity to the embedding of its text caption (e.g., "an aggressive, buzzing synthesizer bass").

To deploy CLAP within a C++ environment, developers must bridge the gap between PyTorch research codebases and production binaries. The LAION-CLAP checkpoints must be translated into the ONNX format utilizing conversion utilities like sklearn-onnx or torch.onnx.export.41 Furthermore, lightweight deployment frameworks, analogous to clip.cpp (which executes CLIP models without massive Python runtimes), demonstrate that large multimodal models can be optimized for bare-metal C++ inference.44

Operationally, the 1-second audio buffer is processed by the local CLAP ONNX model, yielding a high-dimensional vector (e.g., a 512-dimensional embedding).45 Concurrently, the system maintains a pre-calculated matrix of text embeddings representing an extensive array of aesthetic goals or user prompts (e.g., "shimmering pad," "lo-fi percussion," "warm analog brass"). The Euclidean distances (or cosine similarities) between the real-time audio vector and the text vectors are computed.40 The text prompts yielding the highest similarity scores are extracted and dynamically inserted into the LLM's prompt string, providing an interpretive description of the sound without relying on a fixed classification vocabulary.

### **3.2 Texture Resonance Retrieval (TRR) and Wav2Vec2 Analytics**

Despite the powerful semantic alignment capabilities of CLAP, research indicates that it suffers from a fundamental deficiency when evaluating complex DSP architectures: it inherently relies on mean-pooled feature vectors.46 Mean pooling collapses the temporal structure of the audio into a single, static vector. While adequate for identifying discrete sound events (like a dog barking), it obliterates vital information concerning time-variant effects such as frequency modulation, flanging, non-linear distortion, and timbral *texture*.46

To rigorously analyze the intricate textural properties of an evolving synthesizer patch, the Ear module must employ **Texture Resonance Retrieval (TRR)**. Developed specifically to bridge the semantic gap between perceptual user intent and low-level DSP parameters (such as effect chains and LFO routing), TRR eschews single-vector embeddings. Instead, it captures audio texture by utilizing second-order feature statistics—specifically Gram matrices—derived from the deep activations of Wav2Vec2 models.46

#### **Mathematical Formulation of TRR in C++**

The TRR algorithm operates intrinsically on the intermediate internal activations of a pre-trained Wav2Vec2 Base model. Rather than utilizing the final classification layer designed for semantic understanding, TRR intercepts the mid-level transformer layers, explicitly selecting the layer set ![][image1].46

Let the frame-level activation map extracted from a given layer ![][image2] be denoted mathematically as ![][image3], where ![][image4] represents the total number of temporal frames generated by the model's convolutional feature extractor.46

Calculating a full Gram matrix over a 768-dimensional space is computationally prohibitive for real-time applications, scaling exponentially as ![][image5].48 To mitigate this bottleneck, TRR employs a frozen random linear projection strategy.46 A random projection matrix ![][image6] is instantiated once during plugin startup using Xavier initialization (a technique to maintain variance across network layers), and remains completely frozen without further weight updates.46

The activation map is computationally projected to a condensed 32-dimensional subspace:

![][image7]  
Subsequently, the Gram matrix for each layer is computed. The Gram matrix calculates the un-centered covariance, effectively capturing the co-activations between different feature channels—the mathematical essence of acoustic texture.46

![][image8]  
The resulting Gram matrices for the selected mid-level layers are averaged to construct a holistic texture representation:

![][image9]  
Finally, the averaged ![][image10] matrix is mathematically flattened into a 1024-dimensional vector, and an ![][image11] normalization is applied to create a scale-invariant, highly robust texture-aware embedding denoted as ![][image12].46

In a C++ implementation leveraging ONNX Runtime, achieving this requires compiling an ONNX graph customized to output the hidden states of layers 4, 5, and 6 directly.46 The subsequent projection and Gram matrix multiplications are implemented utilizing highly optimized linear algebra libraries such as Eigen, which aggressively exploit Single Instruction, Multiple Data (SIMD) CPU instructions for maximum throughput.50

### **3.3 Computational Cost: The Latency Imperative of Local Inference**

A critical architectural junction involves determining whether to execute these deep embedding models (Zipformer, CLAP, TRR) entirely locally within the plugin host's memory space or to offload the raw audio buffer to a remote HTTP server (or cloud-based inference API).51

The defining metric is interaction latency. The user interaction paradigm dictates that human engagement with an LLM agent adjusting synthesizer parameters requires the perception of immediate, real-time responsiveness.51 A delay in the Ear module's feedback loop causes the LLM to make adjustments based on obsolete audio data.

Empirical research comparing local versus remote audio processing platforms provides definitive answers. In controlled interaction studies, local inference mechanisms achieved an average end-to-end latency of approximately 297 milliseconds (including cognitive processing time).53 In stark contrast, remote interactions involving network transmission protocols, SSL handshakes, and server-side processing yielded latencies averaging 976 milliseconds.53

A delay approaching one full second breaches the cognitive threshold for continuous turn-taking and fluid conversation, effectively transforming a dynamic AI co-pilot into a sluggish, disjointed interface.53 Therefore, the Ear module is architecturally compelled to execute feature extraction exclusively *locally*. To manage the immense computational load within the resource-constrained VST3 environment, the models must be aggressively quantized (INT8) and decoupled entirely from the high-priority audio callback via the SPSC background thread architecture previously delineated.31 Offloading is only architecturally permissible if an external local process (e.g., an independent Python daemon communicating via gRPC or shared memory) is utilized to isolate the plugin from heavy RAM footprint allocations, though this vastly complicates the user installation experience.

## **4\. The C++ Data Pipeline: Audio-to-Text Architecture and CMake Integration**

The synthesis of algorithmic DSP extraction and advanced neural classification culminates in the complete "Ear" data pipeline. The terminal objective is to convert the raw audio buffer into a highly structured JSON string or descriptive text prompt, ensuring it is delivered to the "Brain" LLM agent with zero risk of thread collisions, priority inversions, or memory leaks.

### **4.1 Step-by-Step Architecture of the Data Flow**

The architecture operates sequentially across three distinct threading domains, isolating varying levels of priority to guarantee host stability.

#### **Step 1: Safe Capture (Thread Domain A \- Real-Time Audio)**

The JUCE processBlock function is invoked by the DAW, providing a pointer to an array of floating-point audio samples. Utilizing the juce::AbstractFifo, the audio thread evaluates the remaining capacity in the circular buffer. Upon confirming sufficient space, prepareToWrite yields the precise pointer offsets required to handle continuous and wrap-around memory writes. The incoming samples are rapidly copied utilizing juce::FloatVectorOperations::copy, leveraging AVX/SSE processor optimizations. The finishedWrite method is executed to atomically update the write index, signifying data availability.23 The background thread is subsequently awakened utilizing a lightweight, non-blocking juce::WaitableEvent signal.

#### **Step 2: Parallel Extraction (Thread Domain B \- Background Processing)**

A custom class inheriting from juce::Thread governs the orchestration of the heavy computational tasks. Within its execution loop, it yields to the OS until signaled by the WaitableEvent. Upon waking, it utilizes AbstractFifo::prepareToRead to extract the accumulated 1-second (44,100 samples) acoustic chunk.23

This vector of floats is concurrently fed into two separate analytical branches:

* **Branch A (DSP Analytics):** The vector is pushed into Essentia's RingBufferInput. The data traverses the established DAG, yielding precise numerical features (Spectral Centroid, MFCC arrays, ZCR metrics).12  
* **Branch B (Neural Analytics):** The identical vector is formatted into an Ort::Value tensor. It is processed sequentially through the Zipformer (sherpa-onnx) model to extract discrete semantic tags, and subsequently through the TRR ONNX model to extract the 1024-dimensional texture Gram matrix.29

#### **Step 3: Semantic Formatting and Aggregation**

The background thread collects the outputs from both branches. A JSON object is systematically constructed utilizing a lightweight C++ library such as nlohmann/json or JUCE's native juce::var and juce::JSON::toString utilities.

The resulting structured data payload encapsulates a holistic acoustic profile:

JSON

{  
  "timestamp\_ms": 14500,  
  "dsp\_features": {  
    "spectral\_centroid\_hz": 1205.4,  
    "zero\_crossing\_rate": 0.045,  
    "transient\_detected": true,  
    "mfcc\_mean\_vector": \[ \-320.4, 150.2, \-45.6, 22.1 \]  
  },  
  "semantic\_tags":,  
  "texture\_similarity\_to\_target": 0.85  
}

#### **Step 4: Secure Delivery to the LLM Agent (Thread Domain C \- Network/Application)**

Because the LLM "Brain" relies on asynchronous network requests (whether routing to a local Ollama instance or a remote API), the JSON string cannot be dispatched directly from the extraction background thread. Performing blocking HTTP operations would stall the Ear module, preventing it from processing the next incoming audio buffer and leading to SPSC queue saturation.

To uphold strict thread separation, a secondary lock-free message queue (such as a juce::ConcurrentLinearFifo containing pre-allocated string buffers or command structures) transports the JSON payload from the extraction background thread to the application's dedicated network management thread.19 This Network Thread safely unwraps the JSON and transmits the data to the LLM agent, entirely decoupled from the time-critical DSP operations.

### **4.2 CMake Integration, Build Logistics, and Dependency Resolution**

Modern C++ audio plugin development relies extensively on CMake to manage complex, multi-platform dependency trees. The JUCE framework provides sophisticated CMake helper functions, most notably juce\_add\_plugin, which abstracts the labyrinthine generation of platform-specific VST3 and AU targets.55

Integrating massive external libraries like Essentia and ONNX Runtime requires meticulous CMake configuration, specifically leveraging find\_package paradigms.56

CMake

\# Specify minimum CMake version and standard  
cmake\_minimum\_required(VERSION 3.15)  
project(SoundForge VERSION 1.0.0)  
set(CMAKE\_CXX\_STANDARD 17)

\# Require the JUCE framework  
find\_package(JUCE CONFIG REQUIRED)

\# Define the Sound Forge Plugin target properties  
juce\_add\_plugin(SoundForgeEar  
    IS\_SYNTH TRUE  
    NEEDS\_MIDI\_INPUT TRUE  
    NEEDS\_MIDI\_OUTPUT TRUE  
    FORMATS VST3 AU Standalone)

\# Locate ONNX Runtime (often managed via vcpkg manifest mode)  
find\_package(onnxruntime REQUIRED)

\# Locate Essentia and its myriad dependencies (libfftw3, libyaml, etc.)  
find\_package(Essentia REQUIRED)

\# Link the acquired libraries to the JUCE target  
target\_link\_libraries(SoundForgeEar PRIVATE  
    juce::juce\_audio\_utils  
    juce::juce\_dsp  
    onnxruntime::onnxruntime  
    Essentia::Essentia)

**Crucial Deployment Nuance: Static vs. Dynamic Linking:** Deploying ONNX models in commercial audio plugins presents a well-documented, catastrophic risk regarding static linking. Many modern C++ libraries rely on Google's protobuf serialization library. Attempting to statically link ONNX Runtime directly into the plugin binary frequently results in unresolvable symbol clashes with the protobuf version integrated within ONNX Runtime itself, or worse, with different versions of protobuf used by other plugins loaded concurrently in the same DAW.58

If a user loads two distinct plugins from different manufacturers, both statically linking conflicting versions of protobuf, the host DAW's symbol table is corrupted, inevitably resulting in a total application crash.58

Consequently, the definitive architectural best practice is to compile ONNX Runtime exclusively as a dynamically linked library (.dll on Windows, .dylib on macOS). To circumvent "DLL hell" on the end-user's machine, the dynamic library should be compiled with a custom namespace or prefix, isolating its symbols.58 Alternatively, it can be loaded directly from the plugin bundle's internal resource directory using juce::DynamicLibrary at runtime, ensuring complete encapsulation.58 Package managers like vcpkg can be utilized in manifest mode (vcpkg.json) to automate the acquisition, custom compilation, and deployment of these dynamic dependencies, ensuring robust and reproducible Continuous Integration/Continuous Deployment (CI/CD) pipelines.60

## **Conclusion**

The architecture of the "Ear" module constitutes a sophisticated, highly optimized amalgamation of real-time digital signal processing algorithms and cutting-edge machine learning paradigms. By stringently offloading CPU-intensive calculations to a tightly controlled background thread utilizing lock-free data structures (juce::AbstractFifo), the system adheres to the absolute latency and determinism requirements mandated by VST3 plugin architectures.

The strategic bifurcation of feature extraction into deterministic algorithmic DSP (via Essentia's Streaming graphs utilizing RingBufferInput) and Neural Classification (via Zipformers and Wav2Vec2 layers) guarantees that the LLM receives a prompt that is simultaneously mathematically precise and profoundly semantically rich. Furthermore, by implementing Texture Resonance Retrieval (TRR) through the computation of Gram matrices over frozen random projections, the AI agent is uniquely equipped to mathematically conceptualize complex, non-linear audio textures—an absolute necessity for mastering modern synthesizer control. Deployed securely via rigorous CMake configurations and dynamically linked ONNX environments, this C++ architecture forms an unshakeable, performant foundation for the advancement of LLM-driven audio production ecosystems.

#### **Works cited**

1. An Evaluation of Audio Feature Extraction Toolboxes \- NTNU, accessed May 1, 2026, [https://www.ntnu.edu/documents/1001201110/1266017954/DAFx-15\_submission\_43\_v2.pdf](https://www.ntnu.edu/documents/1001201110/1266017954/DAFx-15_submission_43_v2.pdf)  
2. Is the essentia library implemented in C++ slower than the librosa library implemented in Python? \- Stack Overflow, accessed May 1, 2026, [https://stackoverflow.com/questions/76121336/is-the-essentia-library-implemented-in-c-slower-than-the-librosa-library-imple](https://stackoverflow.com/questions/76121336/is-the-essentia-library-implemented-in-c-slower-than-the-librosa-library-imple)  
3. ESSENTIA: an Open-Source Library for Sound and Music Analysis \- Making sure you're not a bot\!, accessed May 1, 2026, [https://repositori.upf.edu/bitstream/handle/10230/35168/Bogdanov\_21stACMIntConfMultimedia\_esse.pdf?sequence=1\&isAllowed=y](https://repositori.upf.edu/bitstream/handle/10230/35168/Bogdanov_21stACMIntConfMultimedia_esse.pdf?sequence=1&isAllowed=y)  
4. Yuan-ManX/audio-development-tools: Audio Development Tools (ADT) is a project for advancing sound, speech, and music technologies, featuring components for machine learning, sound synthesis, speech and music generation, signal processing, game audio, digital audio workstations (DAWs), and more. · GitHub, accessed May 1, 2026, [https://github.com/Yuan-ManX/audio-development-tools](https://github.com/Yuan-ManX/audio-development-tools)  
5. ESSENTIA: an open source library for audio analysis \- ACM SIGMM Records, accessed May 1, 2026, [https://records.sigmm.org/2014/03/20/essentia-an-open-source-library-for-audio-analysis/](https://records.sigmm.org/2014/03/20/essentia-an-open-source-library-for-audio-analysis/)  
6. Essentia Python tutorial \- Colab, accessed May 1, 2026, [https://colab.research.google.com/github/MTG/essentia-tutorial/blob/master/essentia\_python\_tutorial.ipynb](https://colab.research.google.com/github/MTG/essentia-tutorial/blob/master/essentia_python_tutorial.ipynb)  
7. MTG essentia · Discussions \- GitHub, accessed May 1, 2026, [https://github.com/MTG/essentia/discussions](https://github.com/MTG/essentia/discussions)  
8. Essentia streaming mode architecture, accessed May 1, 2026, [https://essentia.upf.edu/streaming\_architecture.html](https://essentia.upf.edu/streaming_architecture.html)  
9. How to write a streaming algorithm \- Essentia, accessed May 1, 2026, [https://essentia.upf.edu/extending\_essentia\_streaming.html](https://essentia.upf.edu/extending_essentia_streaming.html)  
10. Algorithms reference — Essentia 2.1-beta6-dev documentation, accessed May 1, 2026, [https://essentia.upf.edu/algorithms\_reference.html](https://essentia.upf.edu/algorithms_reference.html)  
11. Essentia Python tutorial, accessed May 1, 2026, [https://essentia.upf.edu/essentia\_python\_tutorial.html](https://essentia.upf.edu/essentia_python_tutorial.html)  
12. How to consume the output of an algorithm in real-time? · Issue \#75 · MTG/essentia \- GitHub, accessed May 1, 2026, [https://github.com/MTG/essentia/issues/75](https://github.com/MTG/essentia/issues/75)  
13. Frequently Asked Questions — Essentia 2.1-beta6-dev documentation, accessed May 1, 2026, [https://essentia.upf.edu/FAQ.html](https://essentia.upf.edu/FAQ.html)  
14. cannot install essentia on MacOS M1 · Issue \#32 \- GitHub, accessed May 1, 2026, [https://github.com/MTG/homebrew-essentia/issues/32](https://github.com/MTG/homebrew-essentia/issues/32)  
15. Tutorials \- JUCE, accessed May 1, 2026, [https://juce.com/learn/tutorials/](https://juce.com/learn/tutorials/)  
16. Low Latency Audio Processing \- QMRO, accessed May 1, 2026, [https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/44697/WANG\_Yonghao\_Final\_PhD\_070918.pdf?sequence=1](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/44697/WANG_Yonghao_Final_PhD_070918.pdf?sequence=1)  
17. Build and install finished successfully, but tests fail on Linux Mint (Rafaella) · Issue \#667 · MTG/essentia \- GitHub, accessed May 1, 2026, [https://github.com/MTG/essentia/issues/667](https://github.com/MTG/essentia/issues/667)  
18. Handling Large Audio Blocks Efficiently at Small Buffer Sizes Without Glitches \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/handling-large-audio-blocks-efficiently-at-small-buffer-sizes-without-glitches/65333](https://forum.juce.com/t/handling-large-audio-blocks-efficiently-at-small-buffer-sizes-without-glitches/65333)  
19. Reading/writing values lock free to/from processBlock \- Getting Started \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947](https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947)  
20. Lock free & real time stuffs (for dummies) \- Getting Started \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/lock-free-real-time-stuffs-for-dummies/58870](https://forum.juce.com/t/lock-free-real-time-stuffs-for-dummies/58870)  
21. Is it safe to run new thread in processBlock? \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/is-it-safe-to-run-new-thread-in-processblock/45613](https://forum.juce.com/t/is-it-safe-to-run-new-thread-in-processblock/45613)  
22. Lock-free queues and visualization of data \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/lock-free-queues-and-visualization-of-data/20659](https://forum.juce.com/t/lock-free-queues-and-visualization-of-data/20659)  
23. How does a consumer of lock-free FIFO communicate back to producer? \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/how-does-a-consumer-of-lock-free-fifo-communicate-back-to-producer/17425](https://forum.juce.com/t/how-does-a-consumer-of-lock-free-fifo-communicate-back-to-producer/17425)  
24. Lock Free Queue data copy between Processor and Editor \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/lock-free-queue-data-copy-between-processor-and-editor/47895](https://forum.juce.com/t/lock-free-queue-data-copy-between-processor-and-editor/47895)  
25. A JUCE-ier way of handling this \- Audio Plugins, accessed May 1, 2026, [https://forum.juce.com/t/a-juce-ier-way-of-handling-this/28876](https://forum.juce.com/t/a-juce-ier-way-of-handling-this/28876)  
26. sherpa-onnx — sherpa 1.3 documentation, accessed May 1, 2026, [https://k2-fsa.github.io/sherpa/onnx/index.html](https://k2-fsa.github.io/sherpa/onnx/index.html)  
27. GitHub \- k2-fsa/sherpa-onnx: Speech-to-text, text-to-speech, speaker diarization, speech enhancement, source separation, and VAD using next-gen Kaldi with onnxruntime without Internet connection. Support embedded systems, Android, iOS, HarmonyOS, Raspberry Pi, RISC-V, RK NPU, Axera NPU, Ascend NPU, x86\_64 servers, websocket server/client, support 12 programming languages, accessed May 1, 2026, [https://github.com/k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)  
28. Sherpa-ONNX Hotwords Biasing Guide | PDF | We Chat | Graphics Processing Unit \- Scribd, accessed May 1, 2026, [https://www.scribd.com/document/795289962/Sherpa](https://www.scribd.com/document/795289962/Sherpa)  
29. Why Onnxruntime runs 2-3x slower in C++ than Python? \- Stack Overflow, accessed May 1, 2026, [https://stackoverflow.com/questions/75241204/why-onnxruntime-runs-2-3x-slower-in-c-than-python](https://stackoverflow.com/questions/75241204/why-onnxruntime-runs-2-3x-slower-in-c-than-python)  
30. Audio tagging — sherpa 1.3 documentation, accessed May 1, 2026, [https://k2-fsa.github.io/sherpa/onnx/audio-tagging/index.html](https://k2-fsa.github.io/sherpa/onnx/audio-tagging/index.html)  
31. sherpa/docs/source/onnx/audio-tagging/pretrained\_models.rst at master \- GitHub, accessed May 1, 2026, [https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/audio-tagging/pretrained\_models.rst](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/audio-tagging/pretrained_models.rst)  
32. How can I use CUDA in the python-api-examples/audio-tagging-from-a-file.py script? · Issue \#1292 · k2-fsa/sherpa-onnx \- GitHub, accessed May 1, 2026, [https://github.com/k2-fsa/sherpa-onnx/issues/1292](https://github.com/k2-fsa/sherpa-onnx/issues/1292)  
33. Integrating Speech Recognition into Intelligent Information Systems: From Statistical Models to Deep Learning \- MDPI, accessed May 1, 2026, [https://www.mdpi.com/2227-9709/12/4/107](https://www.mdpi.com/2227-9709/12/4/107)  
34. c-api.cc \- k2-fsa/sherpa-onnx \- GitHub, accessed May 1, 2026, [https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.cc](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.cc)  
35. sherpa-onnx/sherpa-onnx/python/sherpa\_onnx/offline\_recognizer.py at master · k2-fsa/sherpa-onnx \- GitHub, accessed May 1, 2026, [https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa\_onnx/offline\_recognizer.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/sherpa_onnx/offline_recognizer.py)  
36. Issue with building and running sherpa-onnx gpu on Windows \#878 \- GitHub, accessed May 1, 2026, [https://github.com/k2-fsa/sherpa-onnx/issues/878](https://github.com/k2-fsa/sherpa-onnx/issues/878)  
37. Our first commercial plugin using ONNX \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/our-first-commercial-plugin-using-onnx/58195](https://forum.juce.com/t/our-first-commercial-plugin-using-onnx/58195)  
38. Onnx runtime 1000x slower in c++ than python : r/cpp\_questions \- Reddit, accessed May 1, 2026, [https://www.reddit.com/r/cpp\_questions/comments/yk3300/onnx\_runtime\_1000x\_slower\_in\_c\_than\_python/](https://www.reddit.com/r/cpp_questions/comments/yk3300/onnx_runtime_1000x_slower_in_c_than_python/)  
39. LAION-AI/CLAP: Contrastive Language-Audio Pretraining \- GitHub, accessed May 1, 2026, [https://github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)  
40. CLAP Learning Audio Concepts from Natural Language Supervision | Request PDF, accessed May 1, 2026, [https://www.researchgate.net/publication/371288578\_CLAP\_Learning\_Audio\_Concepts\_from\_Natural\_Language\_Supervision](https://www.researchgate.net/publication/371288578_CLAP_Learning_Audio_Concepts_from_Natural_Language_Supervision)  
41. ONNX Model Converter \- Claude Code Skill \- MCP Market, accessed May 1, 2026, [https://mcpmarket.com/tools/skills/onnx-model-converter](https://mcpmarket.com/tools/skills/onnx-model-converter)  
42. Converters \- ONNX 1.22.0 documentation, accessed May 1, 2026, [https://onnx.ai/onnx/intro/converters.html](https://onnx.ai/onnx/intro/converters.html)  
43. Converting Models to \#ONNX Format \- YouTube, accessed May 1, 2026, [https://www.youtube.com/watch?v=lRBsmnBE9ZA](https://www.youtube.com/watch?v=lRBsmnBE9ZA)  
44. monatis/clip.cpp: CLIP inference in plain C/C++ with no extra dependencies \- GitHub, accessed May 1, 2026, [https://github.com/monatis/clip.cpp](https://github.com/monatis/clip.cpp)  
45. r/AudioProgramming \- Reddit, accessed May 1, 2026, [https://www.reddit.com/r/AudioProgramming/new/](https://www.reddit.com/r/AudioProgramming/new/)  
46. TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2603.09332v1](https://arxiv.org/html/2603.09332v1)  
47. \[2603.09332\] TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2603.09332](https://arxiv.org/abs/2603.09332)  
48. TimberAgent: Gram-Guided Retrieval for Executable Music Effect Control \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2603.09332](https://arxiv.org/pdf/2603.09332)  
49. Audio representations for deep learning in sound synthesis: A review \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/358151666\_Audio\_representations\_for\_deep\_learning\_in\_sound\_synthesis\_A\_review](https://www.researchgate.net/publication/358151666_Audio_representations_for_deep_learning_in_sound_synthesis_A_review)  
50. Build fails with "error: 'av\_register\_all' was not declared in this scope" \#1411 \- GitHub, accessed May 1, 2026, [https://github.com/MTG/essentia/issues/1411](https://github.com/MTG/essentia/issues/1411)  
51. Comparative Analysis of Audio Feature Extraction for Real-Time Talking Portrait Synthesis, accessed May 1, 2026, [https://www.mdpi.com/2504-2289/9/3/59](https://www.mdpi.com/2504-2289/9/3/59)  
52. Comparative Analysis of Audio Feature Extraction for Real-Time Talking Portrait Synthesis, accessed May 1, 2026, [https://arxiv.org/html/2411.13209v1](https://arxiv.org/html/2411.13209v1)  
53. Remote Recording Platforms vs. In-Studio Sessions: A Technical Latency Analysis and London Studio Ecosystem Report \- Finchley Studios, accessed May 1, 2026, [https://www.finchley.co.uk/finchley-learning/visual-podcast/remote-recording-platforms-vs-in-studio-sessions-a-technical-latency-analysis-and-london-studio-ecosystem-report](https://www.finchley.co.uk/finchley-learning/visual-podcast/remote-recording-platforms-vs-in-studio-sessions-a-technical-latency-analysis-and-london-studio-ecosystem-report)  
54. Remote Recording Platforms vs. In-Studio Sessions: A Technical Latency Analysis in London \- Finchley Studios, accessed May 1, 2026, [https://www.finchley.co.uk/finchley-learning/visual-podcast/remote-recording-platforms-vs-in-studio-sessions-a-technical-latency-analysis-in-london](https://www.finchley.co.uk/finchley-learning/visual-podcast/remote-recording-platforms-vs-in-studio-sessions-a-technical-latency-analysis-in-london)  
55. How to use CMake with JUCE · Melatonin \- Sine Machine, accessed May 1, 2026, [https://melatonin.dev/blog/how-to-use-cmake-with-juce/](https://melatonin.dev/blog/how-to-use-cmake-with-juce/)  
56. JUCE/docs/CMake API.md at master \- GitHub, accessed May 1, 2026, [https://github.com/juce-framework/JUCE/blob/master/docs/CMake%20API.md](https://github.com/juce-framework/JUCE/blob/master/docs/CMake%20API.md)  
57. Importing and Exporting Guide — CMake 4.3.2 Documentation, accessed May 1, 2026, [https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html)  
58. Deploying an ONNX model \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/deploying-an-onnx-model/59753](https://forum.juce.com/t/deploying-an-onnx-model/59753)  
59. Our first commercial plugin using ONNX \- Page 2 \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/our-first-commercial-plugin-using-onnx/58195?page=2](https://forum.juce.com/t/our-first-commercial-plugin-using-onnx/58195?page=2)  
60. Dependency Management in ONNX Runtime | onnxruntime, accessed May 1, 2026, [https://onnxruntime.ai/docs/build/dependencies.html](https://onnxruntime.ai/docs/build/dependencies.html)  
61. From Frustration to Seamless Integration: Essential Tips and Proven Solutions for vcpkg with C++ Development \#42579 \- GitHub, accessed May 1, 2026, [https://github.com/microsoft/vcpkg/discussions/42579](https://github.com/microsoft/vcpkg/discussions/42579)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGkAAAAXCAYAAAAIqmGLAAAF8UlEQVR4XtVZW8hnUxRfO0bklrsJzTdy6YtCNKJ4QiNhCiFe5MHggXiQywNJLrnUUOQ+5BIijUfl/+aaJDyQGIkilCKXxvj99trnnL3PXnufc/7zfSO/+n3n/Ndee+119tp77XXOJzIRri8Yg7k6Tce0YVLt4b7DGsQ4LZmgWMG22Kj2zRozwRLAjbA6rLHkqA5Zbfx/oPcIO4Mrwd1S8XaAMZedyGgsg8r7gAeAK3ptk7EneAZ4qlSMTXJvDgT79OU58FfwEfDYTqOMOSfRw+5hSydiR3AtLD0j+jw3S2V+U3Tj74H7B3HdGvG+trUH323A9+HJGkhTTm7F3x/Aw3stFlZB/0Vc9+83DOAC8GTRXUp32P9SSRaE4WUsSpuPBO8BHwXPA3fomlrFU8DfwYuylgoOBt8DbwN3Be8UDdLGWOk/AMefyXCa44p8Ck+6WTQtGjCngav7JUkXJvky9LmLp4A+3AK+Dx4PHgpuAtfHSgHw0dHXG2y3cugDapCYM4mzwS3glY1Si5FGa7BNmNKxQeJu+BusBClGMtbj4CfgN5C/gutZkqz+PlI/o1/rcf8Frgvh95lSWuguCtJIrAW3uGjrie6sQ3hjTt2SIVgvDzImSAuiqeUN8UFyI4KUgCmeK99G2bcOzs8VA7Qh0udO5M46sZV0jfRxs6sGqTPE7f4CBN/iurqVErFzYxydgJK5TO7qQXKaBXhurhEGNE53mTEDqjM5SMbUcIFz11wivhL1C8X0OcAHSapB6nA0+ItouhtwpwTDZfOXhaBRVqwGSfRgvk7UQhuksjkTD4EPgB+CXKxvg8clGsPYIBqk28FnwavBj8C7xa7gJqW7K6RbAS0mPOTF4nP5aH4AHuZ7epRHcpou3sHda6LvSX2sAh8T1aM+A+qDFCuV0I3suEBvlO4cYmX3M7jG9s6UcmzO40y6BcWK9Hvw+vA7hn82xyzmfDYrgo08KH/CwIv9xgamS2LIM4GBqk7byIDwAZ+H6DNcc9+cX533SpLvK0Gqjiu7Sxsgr8jzmDsKx0B9AiPDTZC46BswWDPwU3DfRhi5coLTIN4lOqZZrBwEfi1GOrGfyZYuA/gS/Tr4uWilFQ3c3p4jXZprUA5SgFeOe9iP1JwXX4p+HRgDVohbnVbFDZog/Sb2mbcLeK3oeyA3yxFps+I08B/RF8Z50XyuWSm+rIzpD8/+ffZJpJunbMb4gsm0Y6WL+yVPpX+JrubvQL7U8n0vRxgmXC4TzkH6LtMEqRrwHrhgOPZAkNpn5A59WLgQXZz+U1CbByYfqKR0oOiW56otgefC+TaddX8uuHfoO4QmHc+kXDjEsHYSUwi/IFhnGsGDm5MbguQnMaQ7N5Nu3LodJyfh758uPdtDkFyS7gK4cOnrTT15gqPAH0UdxEur+wrXpzEYVwJfaLna+Sb+pgsH83ZDvKGyEjzbbQ3YwG98PEs4yQ0uF31G+3zRyeVibXc3DF2Iyx+ilWODuh1NXZucpr2m3SgcWv+b3Rqqu/y59gPfQsOTop+B3hX9ukAnYr4qupuCjdzQ0iK1H4ZEkJIVHaHV5y5gWmz8Ztpr0h0XHSec54tVmlPEScR8+ECwlEYhJVeFtgaJnUiuUGcXcWGhw3mlrY9FU9oKY1wfJFcpwVkwHCOpE9zOXIF0Zh0aFrTdML/NiG2m9o3RNsr4dGcY8AKmKL6wFosAp19X1ol+/S9ljkE7ojuShQ9TOwuB3hy2972d1HM9e44qBr5SLwfSAUcGqernatFy3UpTPaRWul/+btBOxYc+uiCN7xQeckyHqk61cRBx7+APg8SvyntFTVPADMGJPT0VZyMZSOQFOwMwTAdR8y5WTHcmDHs9BI1hxRYTVEtgZcmv2/zckpXUnf3iSFyxrCiLCtWmDpEdQz8TZYIYLMxY5LCoWKyryoCpCZjfTjVNNeArwh2i38K0UvKdRvQUQysTLAeMQZzshD9PiP/UJddI92+h0DyIESpS1irJS6jOcSbPBC3KLcPo+qYLZX6bY3qO0RH5F2ww9/Uu8u4NAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAXCAYAAAA2jw7FAAABJ0lEQVR4Xm2SIUxCURSG3x0jsL3tbZDcDOrY2GMjMYsTE8FE02Q1SCTQ2aAatDmD3ZkclZHYywSKxWYwW0x+95x7H/ciZ/zn/P9/zrn3brwkkTBaHLUqcLxSywQt5cbYX9SyeudMv60Z2aBcgNrOXdLvkBawCbgTRx8loyl5jroEp1hDteUKwQ1YgQx5izFw25Iy5gowQ9XAMziWrhvok39t5boexlRb/pW6+UltYb2AZtkjUsSS+gbG4Gq7qXGC/AJr+CPNatCTyS74gb5SM31TdIIZIL8huV0oW46cwTbgA36gXpBJ16Q2tGA3153kEP+JWveT9sZ7ygMYwd/BUXmHu7XC2DmTXWRF3PihLsK/3vf3zUnol/O/bZ3YV8eXOLafgj8xjj9nHBwLi9toMQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAYCAYAAAB5j+RNAAAC90lEQVR4Xq1Wz4tPURQ/N6PETJIfsxjFKGTDYpINkdAoLEipsbJgIcXqKws7OxuyJEUzRbNFUWZBWShSopTIH6CIlTI+n3fue++8d8/9vu9MPvX53vs+55x7z733vPu+Igah3bdCRC05xg74YyaCh7xT3lKi26ONJCIRsmh6JiuO/e7xch7BWHI+fdA3pK9xEDgrdZ8U+8HP4LcsQ+P5QIxz4A2/QJRDBO3eAh+gs9GYboN/wckoLAH3gl/hsZNCbv0Dp5dzLHQ1joIPwXVGWwW+FiYiMlYoimHwPri+PfJCk6t9nPorhaBHdKk0R+wAf4Kz4FChxKTR3EQ7Ujo20Bq7CUftkvBwEs2WyqDWKXAe7LXcV4NnJRnWmeW/wWQVBdbbH3C3EWsk/g46zBZZ13qe6BKy9dZCNeQy9IezE7RRTePBV8UYJsDfYustD9brI/AtuLVlWwS6kzP1pnomhInzLb8IXoXPytJgD6JGIjjIzlWE03ZXGvXmnIPOPormFTrbBps4IhnHlVzk660ZdRichfQD7TSMB6M+Ds5AP4X2DHgP3ASuBW9Av4B2u+gFPx1t9HsCngeXchAHxez5ekuWFI5K048enPwQ+AHcBb4TTfQ0uAf8BJ6Ivj38vke7QfSKeg6Z92uETsg77g34XVhroai3eZh+of0IJyZcBZgcr4DXjDCE/mb4nBMtDX7ueBJsocsk+BQ+/MqEQJ9Q1DfBHXwBjiV70B8xofqH4G5x146XPrFhh4lN2VVE9EK9GCb9TPSkCO7ujLRPa1HQu9AMXmWyRnQHzPEUwKSBb/YxfQwTiHiJMPiHoaBvPRIMPP5xE1cjXayBNQa+oTInmozVmdTj0L5Wij8XYS5wYhW4U9zhwOTQ3gGvi74crTSaE/tQnb/ctSPiHwPra3lLYxTjVhiBcTaWcSOlYzYHRdNqnvaBXyDgGKrrI4E/dlQT4yA74yH15X11GTrvOb5xFep+GqRwFpxzHRzOJyW3CQmchBIhUSv8A5dlXE0XCI3bAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAjCAYAAAApBFa1AAAEw0lEQVR4Xu3cy8vcZBTH8VNaQbFFUcELQkUUVEQpit25EIsFdWFBcKUgeAFxI4IoLaW0LhShINqi1IULQakuBHXhasSFixbBhRsvC0vxDxB1oXg5v/d5Qp7JZNLMZCbJJN8PHCbvk/tlkvOeJGMGoF+2FRuAZXEwAQAAAAAADMWClZ4FBwcwBt2fGLpfAgAAMHijTzhGvwEAbKChn7mGvn71sBUAAAPDpW08trG3sYE4agGU6snJodFiNBp5g4xlPQEAAAaKdA4AkOCyAAAAgLEg9wUADAXXtBFhZwOrdI/Htx6TQnsnBvD13u/xs8c3MSYer6UDNHC9x0GPe+Pfl3i8Gz8xAAM4/pcz2hUHAHTl6fj5ksfFsXtf/KzrihiZ6+LnTx6PWEgCr/X4Jbb/6nFn7MamWEuSspaJAmvEMQugG5fHz4+TtjT5quNSj8Ox+z6Pp5J+8oOFs5yqo6KE7da8NwAALeoi7+5ini1T1ecBjz8t3LK70mO7hYrNfzHUXWmJ7aT5arrZfG+yfL6nLZ9vVpUqoyrSbx474t+6FfhP3rs1z3octbC813jc7fHh1BChGpZ6x+OjQtuFvGflm/oLy9sPeCe3RAEAGKC7PP6w2efWsoStWlkKUV82351J2/tWZ75mb3r8XWjTeFkC14SSnu8tLMvrHo/FtjJpwibanrssXyclna/GbrnFQlK6x+O4heTtNgvz0vNtGvaYTa+HKmva0qq07Y5tSlaVtH5gYR6qsOl26YsWnm/rXLNDA0An+OKiPRxtC2qWsDXTJGGbWHigP6MEp854dXxmoeJXRzFhUxJ1Y+xWgqWESlXEG2KbaJuLxlXiptukJz2etPCiwhuWVxeV4D0eu3V79IXYrSrewxZeElFl9DsL6/+1hYQR6AnOydgQyxyqy4yDVevfXljTEmUJm24nnktiXsJWlsgcsulxizHvua1svuctH/Yvm52v5llMQjTMWQtvRk4sJG9ly1apZJvebotV6ZR0nbCQmP1uoTpWdRv5KsuTMSVmmfvj3+qnlxVuTvq1pGRrrENLs6mrZ4sDAECpRSpsqug8UWhrom6F7RULVazUvx6PWkiOVGFa1XVX66hpFmPe9LMKm97+1DNsl033Tm1NQm92AgDWa945u4c2aFHRqUUStmcsVKCKlHAVE5w05lW+6iRsSoDu8HguadNbkJ9b9UsJy1KlaxHpLdG3PY547J0aAsBKcXkDMDZKeOq+JarE7vnY3ZTmq+nWeUtUv2GmpOiirTHDufoti4nVSk7csxM5VWyokCZsmpIqbXoBoKLSBgDASsxewTBIuv2XJWaKiU0na2mV7YDHg7G7qeJ8swSt2Kbh9GzY7jDaVgKn58TUT8++fRXbV+0hCz+RkVb+irQsL3t86fGjx5nYrpcB9BMcegkAg8T5EQA6wgn4AvZ7fBK75/28xTro4f9PLeygqwv9AADoFukDeir75f42bfcvxMyzanxHeoodAwAAAAAA0BglFgAjwKkOwnEwZOzd/mMfAQAAABgt/iEaIXZ6PWyn9WHbAgCGjOscAAA19fWi2dflAgB0pPrCUN0XANA7nLgBAAAAbAj+fUE7ONIAAP3D1QkAAAAA0Cr+EQUAAABW73+B8K8Sr0BXKgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAyCAYAAADhjoeLAAAFUUlEQVR4Xu3dTchlYxwA8GcaFvIZRaSMQsm3fK5mM4pEQmE9YYoN8pWy0aRECc3IxlhI7GShsJjMgmSjlBKFMooiC3bi+Xvufe95z9z73nvf+3nO+f3q39t9nnPve76ee/73nOc8JyUAAABojx31AhjLXgMAQGNIXgEAAABYN804Z9WMuQQAoEYatyZsCJiddgSLo30xX/YoAGA4WQIAAAAwi6vqBfQ57QIArNb1OW7L8Ve9AhpDTg1AR0jYAADWnIQNIDhjC6wxCRsArJafC4w1bcL2by9+z/HTiOhPU40T4s0ArBu5QgvZqAtyfI6zcpxdiWWZNmH7NJUE7L16RU3sLHty/J3K9Ddurm4S+z0AsBr7chxNg7NlRzZXb6l/1uy6esUWvq0XsGJdyUO7spwsgZ2pe2xzBi7P8U8aJEGv5rgix5PViRZj2zviuTm+SWV+4+zgJHam4f/wxBw/p8Hyf5fjpiTBAzpp2NcksGqR7ESScjDHrl7ZgV7ZEhK2mcQ4bjGf99QrphQJWsSdvdeRrEU/uB83pgAAWJFLcvyW4/N6RXZpWv+ELXyQStL2Sb1iAvEz8uVU3l8XdSVh82MTAKbgwDk3vVX5dirJyqPVugbqX869q14xxrWp3JDwfb1ihLh0OunlV+ZI0weg2WY7kh1OJdGJx0NN48t07DAa/bivMt2yPJfKckSfs/NqdVuJ5Y73Ha6VpxEr9oUcT9ULWS9DtxwAa8a39TQeTiVh2V8piyE2+p3vI5Y5vMcs4gHyMb+/1iu2EMOYxI0Lv1TK3kqV5d8xuCx8b44bcuzuTwi0m8MJLIOWNqnHUklOqmsskpQoq/Rh27RCb81x94i4uDLdMsWZtWkviYa42zSWtXpmLpLU6L9W+rAVMeTI1ZXXAMCxZGCTmXo9xVAXkbC8n+OyXll/YNp533QQMzfNDMa8TeLUHB+n6T676uscf+bY23t9fyr94qoJW1zu3e7nA8D6ad9RrX1L1FdbstNSuUy4KCf1ou6UNLgM+VKlfJLLsXETwBu9v7OKzxj2P2NMurgxI246iHmls9r7XQBAFw0/ro1K2L5IJRl6PpU+ZH3Dkqe6J1I5w7Zor+d4Ji3nf0EDDG/k0F3aBO0xLGG7OZUO/SH6k1WNS9iiz9o0feaerhcAMA+SFWiTesIWg/P+kUa39K0Stq/S5DcZxKO3PsrxTr0CmMGolkv7lG1ti0NH1BO2B3P7j35ro4xK2E7P8WKOc1KZZlhcmOOOHO+mQf+42+PNdJRDDZOwnzSRrTYpa2oeOrEW6wnbD2nooLUbRiVscZPBdgJYF534ygM2aPONUk3YYtMdzXHBoPp/1cuWvYTNVmaR7F8AUFVN2OLB83GZMsY7O5TjjFSG1ag+xH3UGTYAusEvKliBasIWY5v1xd2h0d9sV6UsTJKwxXAg40SDj35vQFs4jAMsTL0P2zjjErZHcjxUL6yJO0njzN2BNFlyR2fM74g/v09qOCuiGWwnYIx5Jmzn5/gwxwO91xflOJjjlbR5bLbPen+vrJQBALAEb+Z4NpXfio+njScQbPrpeFyO/dUCoGucTkpWArBCR9LgyQj70vAvpD1psc9Gnb9hS8Hq2B5QaAuwSo1tgXFZ9VCOk1NJyOLRVtdUJ0hl4V5LZQDdEGfgdg+qAQBYtLiR4MzK60jQ9qZyc8EtqSRzO3vl0Rduc9+5xuaqTMeGBoCOa28y0N4lAwCAKpkvAAAAzeaXLWyb5rNo1jBrZK12x7WaGQAAWk72CQDAJOSNLWJjQmtoztASGjPMro3tqI3LBJ2g8QIAAADQBf8BiWGnSnxkqaQAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAXCAYAAABTYvy6AAAEBklEQVR4Xr1WS4iOURh+31DulygmNMoQpZDLQtIslJGyYWExO7msSJKSRhYW0mxGWaCYJiUpGxbKQliN0izssCBRU4iyIY3nPffr938mPPX83znv7TznO+c75ycyYLatBhRjQqNvN9YzvqaQRkSJcZXONdOIUt/ZUuf/R1FB0dgSDS9Pw9pKvgpUaBqf9v8B/sIQDSWKrqKxjDA0S8sMGarrkBh8t5rhMBc8Dl5F0HmErSYb7XOmgHvAK4b74ZuR1qwPkSMX6KEsTnfkj7SCXqtHrpWgVcFXXg8+BnvBBeAR8AdcJ20AMA28DJ4DV4KHwO/gS7BbApy2VEIFLcNSlLWS1mpqBlq5qNViCPwF7pUO64LPwU/gWhPTB8cjPJeavqAfQ03geR2cGtgDdJhe5I5jOegH7UgrFbVyH36cVpPZDyqtHGgdhFeMB01/DvgU/Eb6DQtOk0k0fcEyFHmP5xtwsRiyRdcN+V0RzSSEti4EZ8UOcoWCxEFwgq1WFq2stXKgVc2HI62UaBXI1lhE+rsQrAO/kN5Ss41tMwqNYcDDpi+DduH3rWFXJC+GvOEz4AkqBLFeqQeUbMMKEq1c0UpjFGol7sJAgdYARpEcHLfAd+CGXGaE7eBP8B44PfGlmIZal/A8RfHk9aQ5nLS4SwNnNmhlr7UZVa2yzW6TLiLbYReK2h1Qgrz5G6Tf9tbEl0skZZMcP3m2K83dxYyCyaCg1e1WD5/vtLJodfZ8gDXgBzhGuP7d7QM/orkzdhpkNZ3BTv4aFbZ3lmZQs5PTSiNU0qqhtJLV2lBMXLKF5DA7mvgEssIvwC2RtaFgAjkvROxZ+pOsDCo10Mpeq69a1kp6BeTQEUrbgO0pPpxok0LPYOsxVhxcLFfF/DBIIP4w0/TlW3wIrgIHyGx7HdfxHVS02huHhwObwGilHtOXQxZXsNa6ifTlLpS2hRSRiQ8FcrqRdFc9FZRHTlj8g1JXoEd5DnbSdnuL+AHEugOvnGbAuVZzQ0Kr+j8hd7yFjBFoVTBaWWldjtRXeN4E54mB5U5lGgXl4NpokpaAT8DPJIcKq4NFOA7eoeofGAeZ9H0KhehZ6slHK1/Fcgq0mli5/0dJH7JlrZ7jGCDSuht8DV4ED5A+dL4au4XdTvJmU14I4nIwzSR9lUWTDiaJE5mP4bnNmwrQCUort9LqybbNRmswuNxtvaz/zO+g+BvqCKlTWi1tK3lq1lZQWqmTVrV/mhTUkERLHWXKqtSm7VHOi9HBreBirJYGcPXDSe3pRFt10iJ1+MjgT3v7dI/i8JMopFPqiXWPoNn7txG+vEkj+AQiFEwtEGfZXTa5Wg3ICqaGTsvY7KugRYoLyWOTV+OfQbMdWgdW8Rs2Mqyo6eho4QAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAYCAYAAAARfGZ1AAACHUlEQVR4Xo1Uvy9EQRDejSMRhYhCREWukUgUl9CchERErb1GNGoUQqLgD5BQiVApNAqNQvgL+BskiFAIheQUxI9vdmffzr7d5b7ctztv5tuZ2X37TikBXQwJmyFt6fD6SOGQS5OoIpGQxp5E5YSr5NAZjcM/jRnkMiTWxlL7MAU+gz+CHwheYO4ttB794Aa4D24iRdUFbDouU6p2AH6Ds4VHRX2PgSiqJzCPgmfKNrOijNT+yugBr8FbBAeKwgZFE50YTmEtgG3WZXZ2BasJuyaWBBgGX8ATsGJdUQt0HHfgm6KuOYxpHQN1v+yl4eKGpu1ptSqdJbRDswPNubKFHGgNJRdrea+MXfATT/W44VDqw8aiXdJuv8DJImRgle68b+Dok7l90kRFAz2OoanszaGdkU8K6Lz1iw7O2wP+DkztbMsddGO6BI/ArlwTDWXPzL6QUETGFjjiHOxFMb0HaxvsFH47ihp0vz/hqBdxH6zCPlQygd0FJV6DsI2ldNtmhMa4/f1WdL8DUBI6S9qZAy2iD2aJbYdFPM2JZ4Oati8E560rXq3pqlHie3DQuiiZnof1DusBNsUcXxXdNMY0lI/K/5fQVXpgku38x8q/ZP6ItPgPYlurJ9hDrCNwn+EUWK0iXhslZdsPISJ/QuPwR6hALlVUJwBH4+0IO1ocOrLhXCDyJxBpEn1lRK1Uye2gtMjnC77YrC0dMqns6xcpy07NvpEJaAAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAYCAYAAABurXSEAAADaUlEQVR4Xp1VT4hPURQ+NzNl/ElCkskYRdkMGhYUK4nEwr+IIguUJFlMKWUWs1AoykLRNGpmwcJC/kQZxZ4FFsrCLCxspLCQjO+79/zeve+9+959z1ffu/ee853zzv3z7hMhjH3mYE3+EdXEkRCG7pK0ZKhB47rSigxhznDyIarsFlX2FBJxJXfE4E0lZx6hOyLNmRLaGrSPbDwBiyqNqXZ5pBWt0H6uFfjPROkwtdom7EegmlDq7ZF+OV+lshFqI6pSlyYWTqAgjBcaosreHpG6vMFbi41HzlDvTbljhjwS7irUhznvAnA/eAOGYZi2oL8MvAheA3tTaVLuEF6aWNaS3WMn+BOcAk+AI+Bv8C+inqGdBq8WM1Tlq7Kn0TxyIfgeAdMI2aM2Ro/jyWJHMTiHdlUWoUi/IqKImKpQJ10CfgZZ9K7APqZFP4J9ZmBXJFbdGlr8EZsgSNGDwUOuNPpn1dUDPtWiT3lpM2i9tShpavVxJz/AH2LPsHzV/i/wPAK6Q6GHJorka7e64W40B2PGwQtgN0aLxd0kWa4g6SLwOngGHID9trjY3kDISR4EH4A3weWwd+K4k/xuJiDdKGE+IwPi860Aj4NPwNPZwhVmB7E9Hl/g4A0yBf9bcS/dCs5Q3RFwM/gR3CsuzZCSQHLDmEPOZzagvSeuAMZ9gHWbuAldEpvPxPK9A/uEi2fkBdo1EsF28I+4qy3Gx+AccKVqeQ1yzJeMojkpDlzhSfURgzr5TZKPmyduV5HP5PIZm08OazxX/BW4VMeU2Oc68Bv4Rpx4X0Yjl9F+F3fWueIMGcKD9zj7PEov0VvNMQxj4u54hWE++FmQGTI5Xwau7IjWMh98Dg6qj4swAXbpOANXiavJM92ZSAcc3RLn3wF2Gbfdu9XPluO14AFxZ/KYpsANZO6L23ZembihkCOfn8WE+Vjsa3H/DucztnAey37VWPTBgTNk+Edk4TwCvLvZDov7M/KD4JZyZSeh79fZ8WV3xa0W/euFV6XIUXHHhrcShS6u8OKO3Xg7C+TxYAyLvgNeEfdN6HT9rHkv80jwy/0k7nfOlmP+cPTaMzxzs10/wyzxHyrBPm+f0MY3FeOIop2FZkfBuBxzM29ul3KDMkJ3qZ+IDdFCKkV1ud5SJQkkNAm3NFFEi9LmH4aje4V2pXgIAAAAAElFTkSuQmCC>