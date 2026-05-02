# **Comprehensive Architecture for Local AI Inference and API Bridging in C++ Audio Plugins**

## **1\. Introduction and Architectural Imperatives**

The convergence of generative artificial intelligence and real-time digital signal processing (DSP) necessitates a fundamental reevaluation of software architecture within digital audio workstations (DAWs). The development of advanced, AI-assisted audio tools—such as the conceptual "Sound Forge," a hybrid VST3 synthesizer plugin executing within the JUCE framework—requires a sophisticated "System of Models." This system pairs macro-generative capabilities (Large Language Models generating parameter payloads, preset interpolation, or code logic) with micro-neural operations (specialized deep learning models analyzing acoustic properties or synthesizing audio streams).

Integrating these disparate computing paradigms into a single unified local plugin introduces immediate architectural friction. The DAW execution environment is characterized by strict, hard real-time deadlines. At a standard sample rate of ![][image1] Hz with a buffer size of ![][image2] samples, the plugin processor has approximately ![][image3] milliseconds to calculate all necessary DSP algorithms and return the buffer to the host application.1 Failure to meet this deadline—due to thread preemption, dynamic memory allocation, mutex contention, or CPU thermal throttling—results in buffer underruns, producing audible clicks and system instability.3

Conversely, modern AI inference operations, whether executing Large Language Models (LLMs) via llama.cpp or deploying vision/audio transformer models via the ONNX Runtime (ORT), require massive memory bandwidth, unconstrained dynamic heap allocations, and multithreaded throughput that aggressively saturates the host CPU.6

This research report provides an exhaustive, highly technical roadmap for resolving this contention. It details the precise CMake linking strategies, C++ memory management paradigms, thread isolation techniques, and HTTP bridging architectures necessary to embed heavy generative AI and deterministic neural DSP into a JUCE audio plugin without compromising real-time performance or inducing user interface (UI) paralysis.

## **2\. In-Process Native LLM Inference (llama.cpp)**

Embedding llama.cpp directly into a C++ JUCE project provides the highest degree of autonomy for the plugin. It removes dependencies on external servers or network adapters, enabling zero-latency communication between the text generation loop and the plugin's internal parameter state. However, statically compiling and dynamically executing a vast generative model within a host application requires rigorous constraints on build systems, memory allocation, and thread prioritization.

### **2.1 CMake Integration and Dependency Management**

The integration of llama.cpp into a JUCE CMake project requires navigating overlapping dependencies, compiler target flags, and conflicting module source groupings. llama.cpp is distributed as a standalone CMake project that defines specific executable and library targets—most critically the llama, ggml, and common targets.8

When utilizing add\_subdirectory() to include llama.cpp within a parent JUCE CMakeLists.txt, developers often encounter symbol collisions and source grouping errors. JUCE's native JUCE\_ENABLE\_MODULE\_SOURCE\_GROUPS property dynamically restructures how source files are presented in IDEs (such as Xcode or Visual Studio). This property interacts poorly with targets that were not created using the juce\_add\_\* functions, frequently breaking the build of external submodules like llama.cpp.11 Furthermore, llama.cpp attempts to invoke find\_package(OpenSSL) internally, which can trigger hard configuration errors if the parent JUCE project or environment has already imported OpenSSL targets.12

To circumvent these build failures, the CMake configuration must explicitly disable non-essential llama.cpp targets, isolate source grouping, and enforce static linking.

CMake

cmake\_minimum\_required(VERSION 3.20)  
project(SoundForgePlugin VERSION 1.0.0)

\# Add the JUCE framework as a subdirectory (or via FetchContent)  
add\_subdirectory(JUCE)

\# Disable the generation of IDE source groups for external dependencies  
\# to prevent JUCE from corrupting llama.cpp target definitions   
set(JUCE\_ENABLE\_MODULE\_SOURCE\_GROUPS OFF CACHE BOOL "" FORCE)

\# Configure llama.cpp to build strictly as a static library  
set(BUILD\_SHARED\_LIBS OFF CACHE BOOL "" FORCE)  
set(LLAMA\_STATIC ON CACHE BOOL "" FORCE)

\# Disable unnecessary llama.cpp executable builds to reduce compile times  
set(LLAMA\_BUILD\_EXAMPLES OFF CACHE BOOL "" FORCE)  
set(LLAMA\_BUILD\_TESTS OFF CACHE BOOL "" FORCE)  
set(LLAMA\_BUILD\_SERVER OFF CACHE BOOL "" FORCE)

\# Add llama.cpp source directory  
add\_subdirectory(modules/llama.cpp)

\# Define the JUCE Audio Plugin Target  
juce\_add\_plugin(SoundForge  
    IS\_SYNTH TRUE  
    NEEDS\_MIDI\_INPUT TRUE  
    NEEDS\_MIDI\_OUTPUT FALSE  
    FORMATS VST3 AU Standalone  
    PRODUCT\_NAME "Sound Forge"  
)

juce\_generate\_juce\_header(SoundForge)

target\_sources(SoundForge PRIVATE  
    Source/PluginProcessor.cpp  
    Source/PluginEditor.cpp  
    Source/LlamaInferenceEngine.cpp  
)

\# Link against JUCE modules and the necessary llama.cpp targets  
target\_link\_libraries(SoundForge PRIVATE  
    juce::juce\_audio\_utils  
    juce::juce\_audio\_processors  
    llama       \# Core inference API \[8, 10\]  
    ggml        \# Tensor math library \[8, 9\]  
    common      \# Llama.cpp common utilities (previously libcommon) \[9\]  
)

\# Platform-specific hardware acceleration flags  
if (APPLE)  
    \# Enable Apple Silicon Metal Framework acceleration  
    target\_compile\_definitions(SoundForge PRIVATE GGML\_USE\_METAL=1)  
    target\_link\_libraries(SoundForge PRIVATE "-framework Foundation \-framework Metal \-framework MetalKit") \[13\]  
elseif (WIN32)  
    \# Enable Advanced Vector Extensions for Windows CPUs  
    target\_compile\_definitions(SoundForge PRIVATE GGML\_USE\_AVX2=1)  
endif()

This configuration ensures that the libllama.a and libggml.a binaries are seamlessly statically linked into the final VST3 plugin artifact, mitigating dependency deployment issues on the end-user's machine.14

### **2.2 Memory Management: Loading GGUF and Instantiating llama\_context**

The execution state of a local LLM relies heavily on meticulous memory management. In llama.cpp, memory is bifurcated into two primary components: the llama\_model, which holds the immutable neural network weights from a GGUF file, and the llama\_context, which encapsulates the transient Key-Value (KV) cache and execution graph for a specific inference session.16

Loading quantized models requires configuring llama\_model\_params. The use\_mmap flag is particularly critical; by defaulting to true, it utilizes virtual memory mapping (mmap on POSIX systems). This allows the operating system to page portions of the model from the disk into physical RAM on demand, bypassing the need to allocate monolithic blocks of heap memory upfront, thus preserving resources for the DAW host.18

Furthermore, constructing inference batches via llama\_batch\_init is a known source of memory faults if misunderstood. The function requires an integer defining the maximum capacity (n\_tokens\_alloc), but the resulting structure explicitly initializes the batch.n\_tokens variable to 0\. Developers must not manually override batch.n\_tokens \= n\_tokens\_alloc; the n\_tokens integer acts as an iterator representing the *current* active tokens in the batch, which starts empty and increments as the prompt is ingested.19

C++

\#**include** "llama.h"  
\#**include** \<stdexcept\>  
\#**include** \<string\>

class LlamaEngine {  
public:  
    LlamaEngine(const std::string& ggufFilePath) {  
        // Initialize backend (discovers Metal, CUDA, or CPU thread pools)  
        llama\_backend\_init();  
          
        // Configure model loading parameters  
        llama\_model\_params model\_params \= llama\_model\_default\_params();  
        model\_params.n\_gpu\_layers \= 32; // Offload specified layers to VRAM   
        model\_params.use\_mmap \= true;   // Utilize memory-mapped files   
          
        model \= llama\_load\_model\_from\_file(ggufFilePath.c\_str(), model\_params);  
        if (\!model) {  
            throw std::runtime\_error("Critical Failure: Unable to map GGUF to memory.");  
        }

        // Configure the context window and KV cache sizing  
        llama\_context\_params ctx\_params \= llama\_context\_default\_params();  
        ctx\_params.n\_ctx \= 4096; // Explicitly size the context window   
        ctx\_params.no\_perf \= true; // Disable runtime metrics reporting to save cycles   
          
        ctx \= llama\_new\_context\_with\_model(model, ctx\_params);  
        if (\!ctx) {  
            llama\_free\_model(model);  
            throw std::runtime\_error("Critical Failure: Unable to allocate KV Cache.");  
        }  
          
        // Initialize an empty batch capable of processing up to 512 tokens at once.  
        // NOTE: batch.n\_tokens is deliberately initialized to 0\. \[19\]  
        batch \= llama\_batch\_init(512, 0, 1);  
    }

    \~LlamaEngine() {  
        // Strict deterministic teardown of resources  
        llama\_batch\_free(batch);  
        if (ctx) llama\_free(ctx);  
        if (model) llama\_free\_model(model);  
        llama\_backend\_free();  
    }

    // Accessors for threaded inference loops  
    llama\_model\* getModel() const { return model; }  
    llama\_context\* getContext() const { return ctx; }  
    llama\_batch& getBatch() { return batch; }

private:  
    llama\_model\* model \= nullptr;  
    llama\_context\* ctx \= nullptr;  
    llama\_batch batch;  
};

### **2.3 The Generative Loop and Background Thread Isolation**

The core generative function of llama.cpp is llama\_decode(). This function executes a forward pass of the neural network over the current batch. This operation is fundamentally synchronous; it blocks the calling thread entirely until the entire batch of tokens has been mathematically projected through all attention and feed-forward layers.21

If llama\_decode() is called on the JUCE Message Thread, the plugin's graphical user interface will freeze entirely, preventing user interaction and DAW repaints. If called on the JUCE audio thread, it will trigger an immediate and catastrophic buffer underrun, as the execution time of an LLM forward pass is orders of magnitude greater than the ![][image4] ms latency limit.3 Therefore, the inference loop must be executed strictly on an isolated juce::Thread or managed via a juce::ThreadPool.23

Thread priority is a critical secondary concern. The background LLM thread must be initialized with a priority strictly lower than the audio thread to prevent OS-level priority inversion scenarios. Priority inversion occurs when a low-priority thread holding a hardware resource preempts a high-priority real-time thread.22 To maintain audio stability, the background thread must be instantiated with startThread(juce::Thread::Priority::lower).

Furthermore, developers must be extremely careful when configuring the thread count for llama\_context. By default, setting a high thread count for CPU inference (e.g., matching the number of logical hyper-threads) creates excessive context-switching overhead and CPU wait cycles, causing performance to fall off a cliff.25 It is universally recommended to constrain llama.cpp thread counts strictly to the number of physical CPU cores.27

The following architectural boilerplate demonstrates a robust C++ class that isolates the llama\_decode loop onto a safe background thread, processes inference, and utilizes juce::MessageManager::callAsync to push tokens back to the UI without blocking.

C++

\#**include** \<JuceHeader.h\>  
\#**include** "llama.h"

class InferenceThread : public juce::Thread {  
public:  
    InferenceThread(LlamaEngine& engine)   
        : juce::Thread("SoundForge\_LLM\_Thread"), llamaEngine(engine) {}

    // Public API called from the GUI thread to initiate text generation  
    void requestGeneration(const std::string& promptText) {  
        const juce::ScopedLock sl(jobLock);  
        currentPrompt \= promptText;  
        jobPending \= true;  
          
        // Launch thread with explicitly lowered priority to prevent DAW starvation \[24\]  
        startThread(juce::Thread::Priority::lower);  
        notify(); // Wake up the thread from wait()  
    }

    void run() override {  
        while (\!threadShouldExit()) {  
            wait(\-1); // Block thread until notify() is called

            std::string promptToProcess;  
            {  
                // Safely extract the prompt using a mutex lock  
                const juce::ScopedLock sl(jobLock);  
                if (\!jobPending) continue;  
                promptToProcess \= currentPrompt;  
                jobPending \= false;  
            }

            executeGenerationLoop(promptToProcess);  
        }  
    }

private:  
    LlamaEngine& llamaEngine;  
    juce::CriticalSection jobLock;  
    std::string currentPrompt;  
    bool jobPending \= false;

    void executeGenerationLoop(const std::string& prompt) {  
        auto\* ctx \= llamaEngine.getContext();  
        auto\* model \= llamaEngine.getModel();  
        auto& batch \= llamaEngine.getBatch();

        // 1\. Tokenize the input string into numerical IDs  
        std::vector\<llama\_token\> promptTokens; // (Assume tokenization logic populates this)  
          
        // Initialize batch with the prompt tokens  
        llama\_batch\_clear(batch);  
        for (size\_t i \= 0; i \< promptTokens.size(); \++i) {  
            llama\_batch\_add(batch, promptTokens\[i\], i, { 0 }, false);  
        }  
        batch.logits\[batch.n\_tokens \- 1\] \= true; // Request logits only for the last token

        // Initialize Sampler Chain (Greedy sampling for deterministic output)  
        llama\_sampler\* smpl \= llama\_sampler\_chain\_init(llama\_sampler\_chain\_default\_params());  
        llama\_sampler\_chain\_add(smpl, llama\_sampler\_init\_greedy()); \[21\]

        // 2\. The core Autoregressive Decoding Loop  
        while (\!threadShouldExit()) {  
            // Evaluate the current batch to generate logits   
            if (llama\_decode(ctx, batch)\!= 0) {  
                juce::Logger::writeToLog("LLM Error: llama\_decode failed.");  
                break;  
            }

            // Sample the next token from the probability distribution   
            llama\_token new\_token\_id \= llama\_sampler\_sample(smpl, ctx, \-1);  
              
            // Check for End of Generation (EOG) / End of Stream token   
            if (llama\_vocab\_is\_eog(llama\_model\_get\_vocab(model), new\_token\_id)) {  
                break;   
            }

            // Convert the token ID back to a string chunk (piece)   
            char token\_str;  
            llama\_token\_to\_piece(llama\_model\_get\_vocab(model), new\_token\_id, token\_str, sizeof(token\_str), 0, true);  
            std::string decodedPiece(token\_str);

            // 3\. Dispatch the generated string piece back to the UI  
            // callAsync guarantees execution on the Message Thread safely \[30\]  
            juce::MessageManager::callAsync(\[this, decodedPiece\]() {  
                // E.g., invoke a callback to update a juce::TextEditor or Obsidian view  
                if (onTokenGenerated) onTokenGenerated(decodedPiece);  
            });

            // 4\. Prepare the batch for the next iteration using the new single token  
            batch \= llama\_batch\_get\_one(\&new\_token\_id, 1); \[21\]  
        }  
          
        llama\_sampler\_free(smpl);  
          
        // Notify UI that generation has finished  
        juce::MessageManager::callAsync(\[this\]() {  
            if (onGenerationComplete) onGenerationComplete();  
        });  
    }

public:  
    std::function\<void(const std::string&)\> onTokenGenerated;  
    std::function\<void()\> onGenerationComplete;  
};

## **3\. Out-of-Process Inference (Local HTTP Bridging)**

Embedding massive neural networks natively into a VST3 plugin introduces risks of binary bloat and memory fragmentation.31 Furthermore, if a llama.cpp deployment causes a segmentation fault, the entire host DAW will crash. To isolate the plugin and distribute the load, architectures increasingly favor out-of-process inference. This design decouples the LLM entirely, relying on the plugin to act as a lightweight client bridging to a local REST/streaming server—such as Ollama or LM Studio—running on localhost:11434.32

### **3.1 Network Stack Evaluation: juce::WebInputStream vs External HTTP Libraries**

To execute API requests against a local server, developers must choose a networking stack. JUCE provides native networking via the juce::URL and juce::WebInputStream classes.34 However, for long-polling streaming applications, the JUCE native stack exhibits profound structural limitations.

Ollama and other modern AI API endpoints stream their text generation responses progressively. This is accomplished using Newline-Delimited JSON (NDJSON), transmitted over HTTP/1.1 with Transfer-Encoding: chunked.35 juce::WebInputStream operates primarily as a synchronous file-like pull stream. Extensive testing demonstrates that WebInputStream::connect() can suffer from massive latency delays due to underlying OS-level SSL/TLS handshake configurations.37 More critically, developers report instances where WebInputStream fails to close sockets correctly upon destruction, leading to "Broken Pipe" server errors and resource exhaustion.38 Furthermore, its read() architecture does not provide an ergonomic paradigm for infinitely blocking, chunk-based streaming reads.39

Conversely, lightweight C++ HTTP libraries like cpp-httplib (by yhirose) or libcurl are explicitly designed for granular socket management and asynchronous data streams. cpp-httplib provides an elegant stream::Post function that binds directly to the socket level, keeping the memory footprint minimal by triggering a callback iteratively as each chunk arrives over the network.33 While libcurl remains the industry standard, it introduces complex C-style memory management overhead, making header-only libraries like cpp-httplib highly favorable for JUCE plugin environments.42

| Feature Comparison | juce::WebInputStream | cpp-httplib | libcurl |
| :---- | :---- | :---- | :---- |
| **Dependency Burden** | Native to JUCE Core | Single header file (httplib.h) | Heavy dynamic library requirement |
| **NDJSON / SSE Streaming** | Difficult; requires manual byte array manipulation | Native via stream::Get/Post with iterative lambdas | Highly supported via write callbacks |
| **Memory Footprint** | Can accumulate large block buffers | Minimal RAM overhead per chunk processed | Highly configurable |
| **Connection Teardown** | Occasional dangling socket bugs documented 38 | Clean socket termination upon request completion | Fully deterministic socket control |

### **3.2 Asynchronous HTTP POST and NDJSON Streaming Logic**

Executing a query to a local Ollama server requires a meticulously crafted background thread. The thread must construct a JSON payload, initiate a POST request to /api/generate, and hold the HTTP connection open.

As Ollama returns data in the application/x-ndjson format, the response arrives as a series of isolated JSON objects separated by newline \\n characters.35 Because TCP streams are continuous byte streams, a single network chunk may contain a fragmented JSON string. A robust client must implement a buffer mechanism to safely accumulate characters until a newline is detected before parsing the JSON.

Furthermore, out-of-process connections demand rigorous timeout and error-handling protocols. If the local Ollama daemon is not running, the plugin must gracefully recover and alert the user without hanging the internal thread structure.

The following boilerplate utilizes cpp-httplib and nlohmann::json to establish a safe, streaming HTTP bridge inside a juce::Thread.

C++

\#**include** "httplib.h"  
\#**include** \<nlohmann/json.hpp\>  
\#**include** \<JuceHeader.h\>  
\#**include** \<string\>

class OllamaBridgeThread : public juce::Thread {  
public:  
    OllamaBridgeThread() : juce::Thread("Ollama\_Stream\_Bridge") {}

    void triggerLocalInference(const std::string& promptText) {  
        currentPrompt \= promptText;  
        startThread(); // Spawn background network operation  
    }

    void run() override {  
        // Initialize client targeting the default local Ollama port  
        httplib::Client cli("http://localhost:11434"); \[33\]  
          
        // Set strict timeouts to prevent thread hanging if the server is offline  
        cli.set\_connection\_timeout(2, 0); // 2 second connection wait  
        cli.set\_read\_timeout(15, 0);      // 15 seconds read wait for large model loading

        nlohmann::json payload \= {  
            {"model", "qwen2.5-coder"},  
            {"prompt", currentPrompt},  
            {"stream", true}              // Request NDJSON chunked streaming \[36\]  
        };  
        std::string requestBody \= payload.dump();

        std::string lineBuffer; // Buffer for handling TCP fragmentation

        // Execute blocking HTTP POST request with a streaming read lambda   
        auto res \= cli.Post("/api/generate", requestBody, "application/json",  
            \[&\](const char\* data, size\_t data\_length) {  
                // If plugin UI is closed, immediately abort the network socket read  
                if (threadShouldExit()) return false; 

                // Accumulate incoming TCP byte chunk into the line buffer  
                lineBuffer.append(data, data\_length);

                // Search for newline characters defining complete JSON objects \[44\]  
                size\_t pos;  
                while ((pos \= lineBuffer.find('\\n'))\!= std::string::npos) {  
                    std::string jsonLine \= lineBuffer.substr(0, pos);  
                    lineBuffer.erase(0, pos \+ 1);

                    if (jsonLine.empty()) continue;

                    try {  
                        auto jsonChunk \= nlohmann::json::parse(jsonLine);  
                        std::string generatedToken \= jsonChunk\["response"\].get\<std::string\>();  
                        bool isDone \= jsonChunk\["done"\].get\<bool\>(); \[35\]

                        // Safely dispatch the token back to the JUCE GUI Thread \[30\]  
                        juce::MessageManager::callAsync(() {  
                            if (onTokenStream) onTokenStream(generatedToken);  
                            if (isDone && onStreamFinished) onStreamFinished();  
                        });

                    } catch (const nlohmann::json::parse\_error& e) {  
                        juce::Logger::writeToLog("JSON Parse Error: " \+ std::string(e.what()));  
                    }  
                }  
                return true; // Return true to keep the stream alive  
            }  
        );

        // Error Recovery Protocol  
        if (\!res |

| res-\>status\!= 200) {  
            juce::MessageManager::callAsync(\[this\]() {  
                juce::Logger::writeToLog("Local API Error. Is Ollama running?");  
                if (onNetworkError) onNetworkError();  
            });  
        }  
    }

public:  
    std::function\<void(const std::string&)\> onTokenStream;  
    std::function\<void()\> onStreamFinished;  
    std::function\<void()\> onNetworkError;

private:  
    std::string currentPrompt;  
};

## **4\. ONNX Runtime (ORT) Integration & Thread Management**

While generative text capabilities interface primarily with the plugin's graphical user interface, acoustic parameterization and audio analysis require dedicated digital signal processing models. Tasks such as audio feature extraction, source separation, or pitch detection are optimally executed using specialized, static neural networks trained in PyTorch and exported to the .onnx format.45 The C++ ONNX Runtime (ORT) API allows developers to load these models natively into the plugin.

However, integrating ORT into an audio plugin introduces the most severe thread contention hazards discussed thus far. ORT is designed as a cloud-first, high-throughput framework; it relies heavily on unbounded parallelization and aggressive heap allocations (malloc), fundamentally violating real-time safety constraints.3

### **4.1 CMake Linking Strategies for ONNX Runtime**

Unlike llama.cpp, which is often built from source alongside the parent project, the ONNX Runtime is massive and complex, making source compilation highly prohibitive. ORT must be integrated by downloading pre-compiled binaries for the target platform (e.g., .dll and .lib for Windows, .dylib for macOS) and linking them dynamically in CMake.47

CMake

\# Define paths to downloaded pre-built ONNX Runtime binaries  
set(ORT\_INCLUDE\_DIR "${CMAKE\_CURRENT\_SOURCE\_DIR}/external/onnxruntime/include")  
set(ORT\_LIB\_DIR "${CMAKE\_CURRENT\_SOURCE\_DIR}/external/onnxruntime/lib")

\# Add include directories to the JUCE target  
target\_include\_directories(SoundForge PRIVATE ${ORT\_INCLUDE\_DIR})

\# Link the platform-specific dynamic libraries  
if (APPLE)  
    target\_link\_libraries(SoundForge PRIVATE "${ORT\_LIB\_DIR}/libonnxruntime.dylib")  
elseif (WIN32)  
    target\_link\_libraries(SoundForge PRIVATE "${ORT\_LIB\_DIR}/onnxruntime.lib")  
      
    \# Optional: Configure delay loading to prevent crashes if DLL is missing   
    set\_property(TARGET SoundForge APPEND\_STRING PROPERTY LINK\_FLAGS " /DELAYLOAD:onnxruntime.dll")  
endif()

Note: A critical deployment strategy is placing the customized ORT build (e.g., onnxruntime.dll) directly adjacent to the VST3 binary, or statically linking custom-built "pruned" libraries, to prevent dependency hell on Windows systems.45

### **4.2 Restricting Session Options and Thread Configuration**

By default, an ORT InferenceSession probes the host CPU and spawns an extensive thread pool corresponding to the total number of physical and logical processors available.49 This architecture is devastating when running in parallel with a DAW. The host OS scheduler will aggressively preempt the DAW's real-time audio thread to service ORT's multi-threaded intra-operator mathematics (like matrix multiplication), leading to immediate audio dropouts.50

To safely sandbox ORT within a plugin's background thread, the C++ Ort::SessionOptions must be strictly curtailed. The API provides two primary threading constraints:

1. **SetIntraOpNumThreads**: Controls the number of threads used to parallelize execution *within* a single mathematical operator.49  
2. **SetInterOpNumThreads**: Controls the number of threads used to execute independent operators in parallel.51

For audio plugins, these parameters must both be clamped to 1, forcing the model to evaluate sequentially on its isolated background thread, thereby preventing the spawning of auxiliary threads that pollute the scheduler space.48

### **4.3 Crucial: Disabling Thread-Pool Spinning**

Even when thread counts are clamped to a single core, ORT employs an aggressive optimization feature known as "Thread Spinning." When an ORT thread completes a task, rather than yielding its execution time back to the operating system using standard conditional waits, it enters an active busy-wait loop (a spinlock) to check if the next task is ready.49

This "Active" wait policy drastically reduces latency for continuous inference streams but consumes ![][image5] of the active core's CPU cycles, even while waiting.48 If the OS happens to schedule the DAW's audio thread on the same physical core, the spinlock will aggressively starve the audio thread of execution time, resulting in guaranteed buffer underruns.26

To explicitly disable this behavior and enforce a "Passive" yield policy that releases CPU cycles back to the DAW, developers must use the AddConfigEntry API to configure the allow\_spinning internal keys.49

C++

\#**include** \<onnxruntime\_cxx\_api.h\>

void initializeAcousticModel(Ort::Env& env, Ort::Session\*& session, const std::wstring& modelPath) {  
    Ort::SessionOptions sessionOptions;

    // 1\. Restrict execution to a single sequential thread to minimize DAW contention  
    sessionOptions.SetIntraOpNumThreads(1); \[49, 51\]  
    sessionOptions.SetInterOpNumThreads(1); \[51\]

    // 2\. Maximize static graph optimizations to speed up single-thread execution  
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT\_ENABLE\_ALL); \[50\]

    // 3\. CRUCIAL: Disable thread spinning to prevent CPU cycle stealing.  
    // This stops the internal WorkerLoop from utilizing spinlocks and forces yielding. \[49, 52\]  
    sessionOptions.AddConfigEntry("session.intra\_op.allow\_spinning", "0");  
    sessionOptions.AddConfigEntry("session.inter\_op.allow\_spinning", "0");

    // Initialize the session with the constrained options  
    session \= new Ort::Session(env, modelPath.c\_str(), sessionOptions);  
}

### **4.4 Advanced Hardware Contention: AVX Throttling and Cache Eviction**

Beyond thread scheduling, heavy background inference induces two critical hardware-level phenomena that threaten audio thread integrity: AVX Throttling and Cache Contention.

Modern neural inference frameworks heavily utilize Advanced Vector Extensions (AVX2 and AVX-512) to execute Single Instruction, Multiple Data (SIMD) matrix operations.9 Execution of dense AVX-512 instruction sets draws immense thermal power. To prevent overheating, modern Intel and AMD CPUs protectively throttle their clock frequencies downward (an AVX frequency offset).55 If an ORT background thread triggers AVX throttling, the CPU clock speed drops globally. The real-time audio thread, relying on a highly deterministic clock rate to complete its 128-sample calculation within the ![][image3] ms window, will suddenly overrun its deadline due to the decelerated processor frequency.56

Furthermore, large inference tensor projections result in Last-Level Cache (LLC) thrashing. As the background thread moves megabytes of data through the L3 cache, it physically evicts the audio thread's lookup tables and DSP coefficient arrays. When the audio thread executes its next sample cycle, it suffers a severe performance penalty retrieving data from main RAM.7 Therefore, models loaded into ORT must be aggressively pruned and quantized (e.g., INT8 or Float16) to minimize memory bandwidth saturation.47

## **5\. Micro-Neural DSP (RTNeural)**

While LLMs and ORT models must be strictly segregated onto background threads, the Sound Forge plugin architecture will inevitably require neural networks that run directly inside the processBlock to synthesize or manipulate the acoustic waveform. Applications such as neural amplifier modeling, non-linear distortion simulation, or real-time wavefolding demand sample-by-sample, hard real-time neural evaluation.59

### **5.1 The Hard Constraints of the Audio Thread**

The JUCE processBlock functions under an inflexible real-time paradigm.3 Operations executed within this block are subject to extreme constraints:

1. **Zero-Allocation:** The operating system's heap memory manager utilizes global locks. Any call to new, malloc, or dynamic vector resizing (std::vector::push\_back) within the audio thread creates non-deterministic delays, destroying audio integrity.4  
2. **Lock-Free Execution:** Thread synchronization primitives like std::mutex or juce::CriticalSection risk priority inversion, where the audio thread blocks indefinitely waiting for a background thread to release the lock.5  
3. **Strict Bounded Execution Time:** The Worst-Case Execution Time (WCET) of the DSP math must be consistently lower than the hardware buffer duration.1

General-purpose ML frameworks like PyTorch, TensorFlow, and ORT violate all three of these constraints, making them useless for direct audio stream manipulation.3 Instead, the architecture must employ specialized micro-neural frameworks.

### **5.2 Zero-Allocation Inferencing with RTNeural**

RTNeural is a highly specialized, lightweight C++ inference engine architected specifically for real-time audio processing.65 It achieves zero-allocation compliance by abandoning dynamic network graphs and heap memory entirely.65

By utilizing extensive C++ template metaprogramming, the exact architecture of the neural network (layer types, dimensions, input/output counts) is embedded directly into the compiler type system via its compile-time API.66 This allows the C++ compiler to statically allocate all required memory for node states, hidden matrices, and weights directly on the stack or as fixed-size member variables of the AudioProcessor object.67

RTNeural supports several computational backends optimized for different DSP scenarios:

| Backend Paradigm | Mechanism | Architectural Best Use Case |
| :---- | :---- | :---- |
| **STL (Standard Template Library)** | Pure C++ loops, highly portable. | Fallback for embedded systems; compilers can highly optimize inline functions. 67 |
| **XSIMD** | Direct SIMD instruction interfacing. | Ideal for smaller, shallow networks to avoid the heavy abstraction overhead of dense math libraries. 67 |
| **Eigen** | Complex vectorized matrix math. | Preferred for large, deep networks (like LSTM modeling) where dense matrix multiplications dominate execution time. 67 |

### **5.3 Implementation Architecture within the processBlock**

To integrate an RTNeural model, the plugin processor must load the pre-trained weights (exported from PyTorch to a JSON payload) during initialization, and then execute the deterministic forward pass directly inside the audio buffer loop.59

C++

\#**include** \<JuceHeader.h\>  
\#**include** "RTNeural/RTNeural.h"  
\#**include** \<fstream\>

class NeuralDSPProcessor : public juce::AudioProcessor {  
public:  
    NeuralDSPProcessor() {  
        // Load pre-trained weights during initialization phase ONLY.  
        // File I/O and JSON parsing are strictly forbidden in the processBlock.   
        auto jsonStream \= std::ifstream("distortion\_model\_weights.json", std::ios::binary);  
        RTNeural::json\_parser::loadJson\<float\>(jsonStream, neuralModel);  
    }

    void prepareToPlay(double sampleRate, int samplesPerBlock) override {  
        // Reset recurrent internal states (e.g., clearing LSTM hidden memory cells)  
        // necessary when playback stops and starts   
        neuralModel.reset();   
    }

    void processBlock(juce::AudioBuffer\<float\>& buffer, juce::MidiBuffer& midiMessages) override {  
        // Hard real-time audio execution block  
        // Guaranteed Lock-Free, Zero-Allocation environment \[4, 5, 61\]  
          
        auto\* leftChannel \= buffer.getWritePointer(0);  
        auto\* rightChannel \= buffer.getWritePointer(1);

        // Process audio sample-by-sample  
        for (int i \= 0; i \< buffer.getNumSamples(); \++i) {  
            float input \= { leftChannel\[i\] };  
              
            // Execute the zero-allocation mathematical forward pass  
            float processedSample \= neuralModel.forward(input); \[65\]  
              
            leftChannel\[i\] \= processedSample;  
            rightChannel\[i\] \= processedSample;   
        }  
    }

private:  
    // Define the neural architecture at compile-time to guarantee static memory allocation \[66, 67\]  
    // Example: 1 input, 1 output, with a single dense hidden layer of size 8  
    RTNeural::ModelT\<float, 1, 1,  
        RTNeural::DenseT\<float, 1, 8\>,  
        RTNeural::TanhActivationT\<float, 8\>,  
        RTNeural::DenseT\<float, 8, 1\>  
    \> neuralModel;  
};

This structural paradigm drastically contrasts with LLMs and ORT. Where generative AI prioritizes maximizing system-wide throughput across parallelized threads with immense memory buffers, micro-neural DSP prioritizes strict, mathematically deterministic latency on a single thread utilizing mere kilobytes of statically bound memory.64

## **6\. Synthesis and Architectural Conclusion**

Architecting a multifaceted "System of Models" within a modern C++ audio environment requires a rigorous, multi-tiered approach to computational deployment. Each layer of artificial intelligence must be mapped to a specific execution domain, governed by the inflexible laws of real-time digital audio processing.

1. **Macro-Generative Intelligence (LLMs):** Massive operations, such as generating code structures or parameter datasets via llama.cpp or local HTTP bridging, are inherently non-deterministic. They must be aggressively isolated onto low-priority background threads to prevent OS-level priority inversion. Utilizing robust libraries like cpp-httplib for NDJSON streaming ensures memory footprints remain bounded, protecting the DAW from catastrophic latency spikes, while lock-free dispatchers safely update the UI.  
2. **Acoustic Analysis and Specialized Inference (ONNX Runtime):** Heavy deterministic models must also reside on isolated background threads. However, native API configurations must be actively manipulated. Developers must restrict multi-threading (SetIntraOpNumThreads) and explicitly dismantle performance spinlocks (allow\_spinning \= 0\) to prevent background threads from causing audio dropout via CPU starvation and AVX clock throttling.  
3. **Real-Time Acoustic Synthesis (RTNeural):** Models altering the acoustic waveform stream directly reside at the apex of the performance hierarchy. By relying on C++ metaprogramming and static memory allocation, frameworks like RTNeural satisfy the absolute requirements of the processBlock: zero-allocation, lock-free, and deterministic mathematical projection.

By meticulously segregating these computational workloads and enforcing robust inter-thread communication protocols, software engineers can successfully integrate vast neural architectures into JUCE plugins without compromising acoustic integrity or host application stability.

#### **Works cited**

1. Reading/writing values lock free to/from processBlock \- \#9 by oortone \- Getting Started, accessed May 1, 2026, [https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947/9](https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947/9)  
2. AI audio plugin Idea, how to deal with buffer size? \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/ai-audio-plugin-idea-how-to-deal-with-buffer-size/65144](https://forum.juce.com/t/ai-audio-plugin-idea-how-to-deal-with-buffer-size/65144)  
3. Does ORT have any real-time safety guarantees, i.e. that it's non-blocking? · Issue \#15303 · microsoft/onnxruntime \- GitHub, accessed May 1, 2026, [https://github.com/microsoft/onnxruntime/issues/15303](https://github.com/microsoft/onnxruntime/issues/15303)  
4. Would a lock due to allocation of memory cause sample data to be affected? \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/would-a-lock-due-to-allocation-of-memory-cause-sample-data-to-be-affected/37121](https://forum.juce.com/t/would-a-lock-due-to-allocation-of-memory-cause-sample-data-to-be-affected/37121)  
5. Best coding practices for audio applications \- 2 questions (both answered) \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/best-coding-practices-for-audio-applications-2-questions-both-answered/32297](https://forum.juce.com/t/best-coding-practices-for-audio-applications-2-questions-both-answered/32297)  
6. APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2506.03296v4](https://arxiv.org/html/2506.03296v4)  
7. LLaMCAT: Optimizing Large Language Model Inference with Cache Arbitration and Throttling \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2512.00083v1](https://arxiv.org/html/2512.00083v1)  
8. llama.cpp/cmake/common.cmake at master · ggml-org/llama.cpp · GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/blob/master/cmake/common.cmake](https://github.com/ggml-org/llama.cpp/blob/master/cmake/common.cmake)  
9. ggml-org/llama.cpp: LLM inference in C/C++ \- GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)  
10. target\_link\_libraries — CMake 4.3.2 Documentation, accessed May 1, 2026, [https://cmake.org/cmake/help/latest/command/target\_link\_libraries.html](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)  
11. CMake linking to juce modules \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/cmake-linking-to-juce-modules/48568](https://forum.juce.com/t/cmake-linking-to-juce-modules/48568)  
12. Compile bug: Use as library into a project · Issue \#20413 · ggml-org/llama.cpp \- GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/issues/20413](https://github.com/ggml-org/llama.cpp/issues/20413)  
13. Using Llama.cpp as a dependency in another c++ project. \#7631 \- GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/discussions/7631](https://github.com/ggml-org/llama.cpp/discussions/7631)  
14. Using external non-cmake library in a cmake project : r/cpp\_questions \- Reddit, accessed May 1, 2026, [https://www.reddit.com/r/cpp\_questions/comments/mdmkzm/using\_external\_noncmake\_library\_in\_a\_cmake\_project/](https://www.reddit.com/r/cpp_questions/comments/mdmkzm/using_external_noncmake_library_in_a_cmake_project/)  
15. llama.cpp: Writing A Simple C++ Inference Program for GGUF LLM Models \- Medium, accessed May 1, 2026, [https://medium.com/data-science/llama-cpp-writing-a-simple-c-inference-program-for-gguf-llm-models-12bc5f58505f](https://medium.com/data-science/llama-cpp-writing-a-simple-c-inference-program-for-gguf-llm-models-12bc5f58505f)  
16. Thread safety · ggml-org llama.cpp · Discussion \#499 \- GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/discussions/499](https://github.com/ggml-org/llama.cpp/discussions/499)  
17. Model Loading \- llama.cpp \- Mintlify, accessed May 1, 2026, [https://mintlify.com/ggml-org/llama.cpp/api/model-loading](https://mintlify.com/ggml-org/llama.cpp/api/model-loading)  
18. new llama\_batch\_init/llama\_batch\_free functions a bit difficult to use and may possibly leak memory · ggml-org llama.cpp · Discussion \#17680 \- GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/discussions/17680](https://github.com/ggml-org/llama.cpp/discussions/17680)  
19. Completion results change with the amount of memory available on a Mac? · Issue \#5314 · ggml-org/llama.cpp \- GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/issues/5314](https://github.com/ggml-org/llama.cpp/issues/5314)  
20. llama.cpp/examples/simple/simple.cpp at master · ggml-org/llama ..., accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/simple.cpp](https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/simple.cpp)  
21. Priority inversion in AudioRecorderDemo? \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/priority-inversion-in-audiorecorderdemo/40795](https://forum.juce.com/t/priority-inversion-in-audiorecorderdemo/40795)  
22. Proper usage of Thread for running various tasks \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/proper-usage-of-thread-for-running-various-tasks/60198](https://forum.juce.com/t/proper-usage-of-thread-for-running-various-tasks/60198)  
23. FR: Thread-Priority vs Efficiency/Performance Cores \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/fr-thread-priority-vs-efficiency-performance-cores/49025](https://forum.juce.com/t/fr-thread-priority-vs-efficiency-performance-cores/49025)  
24. llama.cpp and thread count optimization : r/LocalLLaMA \- Reddit, accessed May 1, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/14djns5/llamacpp\_and\_thread\_count\_optimization/](https://www.reddit.com/r/LocalLLaMA/comments/14djns5/llamacpp_and_thread_count_optimization/)  
25. Multithreading support in llama.cpp is probably still pretty busted, assuming it... | Hacker News, accessed May 1, 2026, [https://news.ycombinator.com/item?id=39891055](https://news.ycombinator.com/item?id=39891055)  
26. llama.cpp and thread count optimization \[Revisited\] : r/LocalLLaMA \- Reddit, accessed May 1, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/14jk108/llamacpp\_and\_thread\_count\_optimization\_revisited/](https://www.reddit.com/r/LocalLLaMA/comments/14jk108/llamacpp_and_thread_count_optimization_revisited/)  
27. Examine multi-threaded performance patterns in llama.cpp \- Arm Learning Paths, accessed May 1, 2026, [https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama\_cpp\_streamline/6\_multithread\_analyze/](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/6_multithread_analyze/)  
28. How to Enable Multi-Threading in Llama.cpp? | by Norman Ryan \- Medium, accessed May 1, 2026, [https://medium.com/@llama.cpp0/how-to-enable-multi-threading-in-llama-cpp-1b75c04f467d](https://medium.com/@llama.cpp0/how-to-enable-multi-threading-in-llama-cpp-1b75c04f467d)  
29. Learning-based Memory Allocation for C++ Server Workloads \- GitHub Pages, accessed May 1, 2026, [https://abelay.github.io/6828seminar/papers/maas:llama.pdf](https://abelay.github.io/6828seminar/papers/maas:llama.pdf)  
30. Which js library do you use to work with your local LLM server? (Trying to decide between openai and ollama js libraries, or just using raw HTTP requests \- are there more options out there?) : r/LocalLLaMA \- Reddit, accessed May 1, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1huft1v/which\_js\_library\_do\_you\_use\_to\_work\_with\_your/](https://www.reddit.com/r/LocalLLaMA/comments/1huft1v/which_js_library_do_you_use_to_work_with_your/)  
31. cpp-httplib/README-stream.md at master · yhirose/cpp-httplib · GitHub, accessed May 1, 2026, [https://github.com/yhirose/cpp-httplib/blob/master/README-stream.md](https://github.com/yhirose/cpp-httplib/blob/master/README-stream.md)  
32. juce::WebInputStream Class Reference, accessed May 1, 2026, [https://docs.juce.com/master/classjuce\_1\_1WebInputStream.html](https://docs.juce.com/master/classjuce_1_1WebInputStream.html)  
33. Streaming Responses \- Ollama \- Mintlify, accessed May 1, 2026, [https://mintlify.com/ollama/ollama/api/streaming](https://mintlify.com/ollama/ollama/api/streaming)  
34. Streaming \- Ollama's documentation, accessed May 1, 2026, [https://docs.ollama.com/api/streaming](https://docs.ollama.com/api/streaming)  
35. WebInputStream::connect() very slow performance \- General JUCE discussion, accessed May 1, 2026, [https://forum.juce.com/t/webinputstream-connect-very-slow-performance/21427](https://forum.juce.com/t/webinputstream-connect-very-slow-performance/21427)  
36. WebInputStream not cleaning up correctly on Windows? \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/webinputstream-not-cleaning-up-correctly-on-windows/61637](https://forum.juce.com/t/webinputstream-not-cleaning-up-correctly-on-windows/61637)  
37. juce::InputStream Class Reference \- JUCE Docs, accessed May 1, 2026, [https://docs.juce.com/master/classjuce\_1\_1InputStream.html](https://docs.juce.com/master/classjuce_1_1InputStream.html)  
38. \[Juce 6, macOS\] Reading a WebInputStream using an AudioFormatReader doesn't work · Issue \#741 \- GitHub, accessed May 1, 2026, [https://github.com/juce-framework/JUCE/issues/741](https://github.com/juce-framework/JUCE/issues/741)  
39. AudioFormatReader from URL InputStream has no samples? \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/audioformatreader-from-url-inputstream-has-no-samples/47143](https://forum.juce.com/t/audioformatreader-from-url-inputstream-has-no-samples/47143)  
40. cpp-httplib vs libcurl | LibHunt, accessed May 1, 2026, [https://cpp.libhunt.com/compare-cpp-httplib-vs-libcurl](https://cpp.libhunt.com/compare-cpp-httplib-vs-libcurl)  
41. Why use external libraries (like libcurl) vs. sockets for sending HTTP requests?, accessed May 1, 2026, [https://stackoverflow.com/questions/58943165/why-use-external-libraries-like-libcurl-vs-sockets-for-sending-http-requests](https://stackoverflow.com/questions/58943165/why-use-external-libraries-like-libcurl-vs-sockets-for-sending-http-requests)  
42. olilarkin/iPlug2OnnxRuntime: ML Audio plug-in example ... \- GitHub, accessed May 1, 2026, [https://github.com/olilarkin/iPlug2OnnxRuntime](https://github.com/olilarkin/iPlug2OnnxRuntime)  
43. Machine Learning Audio Plug-ins with iPlug2 and ONNX Runtime | Oli Larkin (Ableton), accessed May 1, 2026, [https://www.youtube.com/watch?v=t662qg12f\_Y](https://www.youtube.com/watch?v=t662qg12f_Y)  
44. Our first commercial plugin using ONNX \- Audio Plugins \- JUCE, accessed May 1, 2026, [https://forum.juce.com/t/our-first-commercial-plugin-using-onnx/58195](https://forum.juce.com/t/our-first-commercial-plugin-using-onnx/58195)  
45. Abnormal performance of SetIntraOpNumThreads(1) · Issue \#2741 · microsoft/onnxruntime, accessed May 1, 2026, [https://github.com/microsoft/onnxruntime/issues/2741](https://github.com/microsoft/onnxruntime/issues/2741)  
46. Thread management | onnxruntime, accessed May 1, 2026, [https://onnxruntime.ai/docs/performance/tune-performance/threading.html](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)  
47. \[Performance\] Slowdown from multiple inference sessions in serial. · Issue \#17011 · microsoft/onnxruntime \- GitHub, accessed May 1, 2026, [https://github.com/microsoft/onnxruntime/issues/17011](https://github.com/microsoft/onnxruntime/issues/17011)  
48. OrtSession.SessionOptions (onnxruntime API), accessed May 1, 2026, [https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.SessionOptions.html](https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.SessionOptions.html)  
49. Run ONNX models using the ONNX Runtime included in Windows ML \- Microsoft Learn, accessed May 1, 2026, [https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/run-onnx-models](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/run-onnx-models)  
50. How to pass model settings to Session.Run() in C++? · Issue \#5751 · onnx/onnx \- GitHub, accessed May 1, 2026, [https://github.com/onnx/onnx/issues/5751](https://github.com/onnx/onnx/issues/5751)  
51. AVX-512 throttling: heavy instructions are maybe not so dangerous \- Daniel Lemire's blog, accessed May 1, 2026, [https://lemire.me/blog/2018/08/25/avx-512-throttling-heavy-instructions-are-maybe-not-so-dangerous/](https://lemire.me/blog/2018/08/25/avx-512-throttling-heavy-instructions-are-maybe-not-so-dangerous/)  
52. 265K mystery throttling resolved by AVX offset, doesn't appear in MSR\_CORE\_PERF\_LIMIT\_REASONS \- Intel Community, accessed May 1, 2026, [https://community.intel.com/t5/Mobile-and-Desktop-Processors/265K-mystery-throttling-resolved-by-AVX-offset-doesn-t-appear-in/m-p/1662018](https://community.intel.com/t5/Mobile-and-Desktop-Processors/265K-mystery-throttling-resolved-by-AVX-offset-doesn-t-appear-in/m-p/1662018)  
53. Avoiding AVX-induced frequency mitigation by fine-grained scheduling of MxKernel tasks \- ESS, accessed May 1, 2026, [https://ess.cs.uni-osnabrueck.de/teaching/theses/MA2021\_Luetke\_Dreimann.pdf](https://ess.cs.uni-osnabrueck.de/teaching/theses/MA2021_Luetke_Dreimann.pdf)  
54. HyperThreading, AVX Offset, Worsening Thermal Throttling \- Flight Sim performance, accessed May 1, 2026, [https://forum.openmr.com/t/hyperthreading-avx-offset-worsening-thermal-throttling-flight-sim-performance/26322](https://forum.openmr.com/t/hyperthreading-avx-offset-worsening-thermal-throttling-flight-sim-performance/26322)  
55. OS-Level Challenges in LLM Inference and Optimizations \- eunomia, accessed May 1, 2026, [https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/](https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/)  
56. jatinchowdhury18/RTNeural-example: An example project ... \- GitHub, accessed May 1, 2026, [https://github.com/jatinchowdhury18/RTNeural-example](https://github.com/jatinchowdhury18/RTNeural-example)  
57. Neural Network Audio Plugin JUCE Demo \- YouTube, accessed May 1, 2026, [https://www.youtube.com/watch?v=Nf-iEypmo1A](https://www.youtube.com/watch?v=Nf-iEypmo1A)  
58. How to avoid allocation of dynamic objects on audio thread \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/how-to-avoid-allocation-of-dynamic-objects-on-audio-thread/23149](https://forum.juce.com/t/how-to-avoid-allocation-of-dynamic-objects-on-audio-thread/23149)  
59. Basic questions: process block, dynamic resizing of things, etc \- Audio Plugins \- JUCE, accessed May 1, 2026, [https://forum.juce.com/t/basic-questions-process-block-dynamic-resizing-of-things-etc/42875](https://forum.juce.com/t/basic-questions-process-block-dynamic-resizing-of-things-etc/42875)  
60. Question about memory allocation in processBlock \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/question-about-memory-allocation-in-processblock/6322](https://forum.juce.com/t/question-about-memory-allocation-in-processblock/6322)  
61. Real-Time Inference of Neural Networks \- The Audio Developer Conference, accessed May 1, 2026, [https://data.audio.dev/talks/2024/real-time-inference-of-neural-networks/slides.pdf](https://data.audio.dev/talks/2024/real-time-inference-of-neural-networks/slides.pdf)  
62. jatinchowdhury18/RTNeural: Real-time neural network inferencing \- GitHub, accessed May 1, 2026, [https://github.com/jatinchowdhury18/RTNeural](https://github.com/jatinchowdhury18/RTNeural)  
63. "Low-Latency Inference of Optimized AI-DSP Models for Hard Realtime Deadlines" by Christopher Johann Clarke (Singapore) | Ircam Forum, accessed May 1, 2026, [https://forum.ircam.fr/article/detail/low-latency-inference-of-optimized-ai-dsp-models-for-hard-realtime-deadlines-by-christopher-johann-clarke-singapore/](https://forum.ircam.fr/article/detail/low-latency-inference-of-optimized-ai-dsp-models-for-hard-realtime-deadlines-by-christopher-johann-clarke-singapore/)  
64. RTNeural: Fast Neural Inferencing for Real-Time Systems \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2106.03037](https://arxiv.org/pdf/2106.03037)  
65. Real-Time Neural Network Inferencing for Audio Processing | by Jatin Chowdhury, accessed May 1, 2026, [https://jatinchowdhury18.medium.com/real-time-neural-network-inferencing-for-audio-processing-857313fd84e1](https://jatinchowdhury18.medium.com/real-time-neural-network-inferencing-for-audio-processing-857313fd84e1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAXCAYAAACxvufDAAAEjklEQVR4Xq1XTchWRRQ+BxWUvhIxsihJpR9aBIpmGEUbkyI0KYOwqCDMiDBS8KMwEMJFURuLFhVFiwq1jQujv4VRRFSboD/6QYsgMmqVEUQ/zzNn7sy5M3Pf737oA887c8+cOXN+5s7cV8RB/cMAgs4YxRlQmagEp4Jo7JScnTjPD05UTGhreUfHoK+Yn0YbyGhP0RFJizoVhpwbwswafXT6E+d558YE4zGsWI1UghkwpF/Je4Lew0LwJfDyxqxLwSfA58A7wDP6w6Ph7WzFMgsoLFY7C3wQUuo8Cp7XHw7glCvB/eCz4I3gnDRSIsrQ6MP4OYne6kLzFvAgRJehvQDcgf4naC/0Shl+bmXnS4hWQj6Fkcfw/I5YcjvQ5mfQ2Qad+ejfAH4Drg2jZk7R7Eb7HrgcXAy+Ipa4eUGjRHSDWfldGKQwyIQl4JvC4LK/7D2DZm96aqEvXwp+C97uZItish6Iz3PBF8DXY7/DPjEfQtVhmP79Al6dVWQF+AN4vZNlqGXyebFMlEGy/zV4cSeIvk+LOdRHGOxXMj4xuNI2h1ABPYp2SszRn8Vse9wsvbm6Ty0gv43PBN8Xe920zDBLvxMybiUaLx3pFv4OvJaCmJS3wU1ZrTBai/julLaJl8Xsc5314L+SgkwJ2gj+J5YobuEj4oKMOlMxWdwZi0yUwb3+lNheroMM6+hutUVAZaYOg7vC6HggGD2p7SBtTU3BpCAjvJwVPyopyKQTgtS6wqEi3KLL4nMdpIGnFt+LECgM/Yr2OimCDA9ekvvRMa1tqwvS1ndBJiBI5brT0GcADKQMphkkXXgIvDWpibaChJ7yYOD2vEqsinTkH/BOp1eXNQt43bwruWIZMUi1NXdKF2TfmK8kD8LvpRlkXeFVkrdpJ5vWOkhW7HOx05FgVbeK6X0ltqg0QiwRgpGyknEbR3mxXRPa29Wq2qEI0nAf+GPBP8SMnRA7qc6RcIIqD42EGA6Pal453YnXDQ+B270O0ir5k9j9yyvhb2kGqfSLpyyvFl4xzUqq+c2TNqL2q7Vd6QSP7IjUg1P6KVp+IFDOCjMpPP1q0xpOYm7x9W5wPvo8KUnOOx88rnYSe2wHf5O0VvDTP9P+2fj9Quq5DrbwI+Cf4BVuhM5x8jIno/bdaF6VfGnfI7YLXnMyj8Xgx+BeJ7tIrIq3xWfYVX4FfST5K4iv00HIvV2bp2kecQ10uAPX1Rk28H6igl0TGjL+oVhlWCF+QjFze2CId9Uh8C30z43zCb43f4kdCq1vTWINeEzM3haxO+1xcJ7zayH6b6A9IPZJ9yL4gdh2jgja3Lq0dS94l/BzUeT+bnAChsbDZclKMJAt6F8ShDW45Z6WdBgRlRpP2g3gZsmHmUNYi4nlK8NE8DS2D+8a8Ek3qvnFfn+0WnoIDcUg8vLYV34sqzyJ3lyv0zCRURnz6P8XHdKaNdqG2lLCjTDbCDBcOacXjeUr0ZiEjsJkC3wPb5KGViVoIOv4P+hDf9a9oD9YqdYYnlyJGsNjcBpMyOxmttM0A2arHxGnzXZ2pV8JhvE/DdDM1pEnBqkAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAYCAYAAAC2odCOAAAEhElEQVR4Xr2XXchmUxTH12q4MDNIRC4mQ5NSImmIqBlJIylpEo0LccGFiEkvSriSyOSziVJygRJuyFeZcuEjccetV4MLmSkZNTTG/7/XPvvj7L3P2c/7vuZX/+fss9Y6++y9ztr7nEckQdOTBj0xER+92EVV0i5ie636r/RTMZVMezPSQRcTWBR3WT7CVk8te8FMPy0a8ekkc1r2eVY4woK0gxV2VrmsYuqnZ0jRnie3FZ/SHVMJbNkjbWfh8YaWvURlO35/g456fQidMDhx3UlofJz4qXegDc2b9dCa9Xyf66CboM0j+8CZ0CPQy9Dj0Jbc7VD0fymOz0EvQddDx0dvaEXUzOz0b+gwdFk+WvdzI/S2DAksJjkzu5bdU3SXw3tyIs9CPyPwEI4X5yGOS9DFJ+jnSrQvhN5Hmw91t8TeeXwAehE6y+sp6APoZB9T5RToNehesUp5QZJZ+8b90C5rJszkZpJKjou2wSRdC22DHoVqSWLMe9DtYtVGToW+ljz+fOhd6ER/Tngtk3RzYhujzPoesVL9AdoPnZ0EHAe9IvZ0pDYLI7e3onqYuHZJ6kni2Jdx5R8Sxul4WNyDVz5kwor8Vi2BKSySISYhVgEr5E5vfUysmu5OYk4T64QV1yROrD3FtqdGJenaTBL3FC7Hj8QlLFzL+KNqR3IF9I9YhZ3rLCqb8fsVdJGPqaDyjGTlqAfR6ZcS1yg7ft63K6zuM6InJsGSpEWSMnyfXAHcR4/AsM27mEzuvywE7sGYu34utufKaDThZNiPWC2EHb8B/Qvt8FGssnI/KnDRfJt8A/20gG4J45nMmHMu4VCrpBp8gzGWSYlvL5H10Fvi39ZqW8wFiX+M349cMYSK2IE2k8RkcVNL9qM2k/PzFDHduQmH+nIrO+Aq+BR6Hc4NiZ3Jetrscjn0nViyDqmtmITYabofDfAGXG4HEXiVLLQf1cY7ReujtJU9XdJakrJ4ZSL2iltKwzdf4C7oM2ijP4dfHxQuyewb0eFLx/aarYlj4FaxUvwexyfHzgn4+j1d7G3jpUm7qmHAniIzAR1XUhk6JOghaJ13nwddI/wAtuoaFwXhi2pZbTwZcT8qHpyeIbZWWYq7yrE04UCug3Zm0tG56E6N55xETvuGSJL/mExjrM1ffjjeFywGk8KNmQ9jn4b9NatiFsoXEvfmAJcS/2Ksr45K3efA78JJVNz/L2UGPKykv8RNKovhyW3et1/DS0F5PCBxv2ES90n+dc3qf0Ls8ydwNfSnxP9ih8WefsTdX/l1ynXq9qOePPXErABW55vQAQ1jVh5/Edt3CJfJsvmifPyv0Dl+bNxz9qL9I473QHeILcFXxe7TSTHT3FB/xq0NOKVuDcy414Zwk03QDV5s99AeYdsjM862O03uyLAA7Y/Y1bOacVWYr6CU+YhjRjGUwlCnM6zNZAfp05kMzGN8O5oWeyykL2pgiC5udmwp7lwYqqbIpDOlO3Cl9Nyg/VSLq72hsDvq1ipFaGFo0xruAl10sdb95Sw68jx+/opKROWW/wGZ26yPYEmXqQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAADU0lEQVR4Xo1VTYjPQRh+39iQjcOKVruJpOTgoEixWaEcVpKiFNoLOZDdtO2W4iA34SC0JTm4CAerfBw2OUhSigs5/Ddyc9hatcnH88z3zG+sfdrnN/N+zbwz877/FQnQODVzDZpocTP1szSmhuDoVsws/w+PqHhWVERIrLSrt5QGo6wp7CLR1vCaAYVvJs5ynXCS6ljOm8itM/tm68/g2gnbWTjcxPw8uLp0COB1Ny9hEYZTIONHxMq5h43chPEqxGsYe8qVNuINn2HcCq4Hx8A/4KCUl5YgUW0HP4GHoeuE5QzG++CC6CILwVvgAwSuxbgOfAPu9A50fgj2I3CO03WAr8EpcEO80WTrOOWCX8EDTl4GfgZbYpIyoPdl8AW42OkG1B52yMnGmUGTYm4j7DDiHAe8ogTsczGMYvwAYYlVmmwPgf1i7QQPMwXDMScTXeAFcJVXtIFXwCdYxZ1A+TcEZhnnMAmvxPcL5vcwnwcuBTu02avn4PMT3IJ5u9h6nE9D6ViCJ+Hiv8BtqYF7hGCVHfj+Bh9hfkPs7d0FX4Jdzo8bsubwzHoRo/FT1pTKCa5i/PyiRWasbNYHO6AtN1KwxLdP7POZWrJu2qYsSJGnYk9Pjqv1G5X4XL1iy2GPk/MksDyL6Tl4R1npGrdOnDz6xD4fT2yu2tn5nLwp3lg75PFE9mB3tVBSeawD6+U6Ai9J1nrurMW1AbvFnvQ2hcSc1hfblgezHRj8TD22HH13GdgkRIYltLGy33eZWbJNMudvziREk0hyd0wAiagvdD5JSMTBJaJZIowexOe0m3uw3fYlMp9tucTLofxKWNga3p6IT2MdD4LTartG3Bb+RsY06aCj4A81rSgTUE248TtGF2x+5N6B0+BmHwgcBz9C6nYaFndarATb+r3kPwW94orVn9xkpvatDTl38jeJPzhc9LGYTWWF03Fr/9P9FjwibF81CQcfB/4b4EHRwnrSHdr+C/GZmOUoREXTVoM32aXWgPvBHihMy1fCkLSy5vbC1l2rfskySQbvasdKYEWVIjPXfHNdxaOiylDaQ/KlwSI9VFCUOis1z1wumbhFTcWpfuNENFT24dBMwsJrmpYS//QoDaVM1HQZ6gcO39nCxzSjouYvs5B/QkPkexwAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAXCAYAAACvd9dwAAAEIklEQVR4Xr1YTYgWRxCtJlkwuP7hYrJEcf3HeDSJCBtBUVFEEfTgst7EHxQSopBACCQQgqwnUTBEJCIeFBFFxJuHRQVFJRFZc0jIQVmQeFAQBEVQ3+vq7unp6Znv23X1wftmuqq6uqq7q3t2RWIYx6jJn0hUhTFZfU5albwjjN1AhadanzlFSdaej7KuqdUWXJeGnrGqwawWo+lThxHF0tLgPcPUBuQVtQY5tDBORkusPwDXQXoET3Iz3j8K2orrjMDIEjwPifZfZjJGwALwAHgU3CQ6boJct5Gi8NEBHgZ/AueA28Fn4BA4M1jVYzx4HDwPLgQXgbfBVdEgHONH8BZEi/GcDV4Ed1ktzZpyquqqkoCyag14GcJPI9lW8DV4DPwwkqegp4PgFXCSk+2FlH2/D1aaxL9gj2uvFevfnAgWCTrBqZKGauFE+hgXKRLYzcMgbCKRo+ngMNr/4fmxN82Aq/AMup2RjH1/Fbs6ttMMYWLGblkPTgRXklu5BCpOiwZEPoKLPsnv3wn42ZPI0tbneNwBd0TibvC+Y7f2yWb3M/gS7DU62exnJzOy3iIaZ7/T0Ya2FbDPgGhdcB8Ts8Bz4A2oP/OGzj2L9odCViAXbtTuFQ2adTTOLnBqrIFeEq3P/eDvwi1pt5/ZLYU7rthrtH6B6KToZHMyB0yRg8UUGHwrrmPxY39X4Jf7+i/R02hQbMJ2m2RQjdahAyoeEE/AL70wY90J4aCE2jS+NpeDT8ENrs8JZ0Nbv2LzwIfgPte24PHcw5eQU3lUzsQK0dXisyMNKrRTRQGu9kOs1MpUkYCBDoKvwNjWb2muKlfXJxfXpe97D+wqQtE31gnrjivzDTgx6Mug83WpsD4vu1J/wuKLRisFrwCcsnZb8mDx0OSM1ism6Zjo6bk+WJiQXNRXx9sA/gN+LVqsnJn/wT5TPVR4UmVrLgMmdg2c69rcZrwSJgeLKrgd88mFw8jWYUjOTVkmOc6WsTc8rwC1VGvW1QXwJppfiTrl86p7BvMygoSX9Vl8ZsSXdpdo7fLEJThxPRJdLUYn94XoAeThkjN+Wy5VG9MfjR9tS8NxLHjnxEd2gFs1fkJdR4N1MGS07jIHXTErwCfCS9jIYzwfRHwEnjHFJb5NdAVOiZcZmSY6TnxhL5fygcJzgl8j9oNAZUYPFFM+UByq4b4FEJjxd2ZEK+Nl7MFt9Rz8TsoBcAKHRa8DlgrfGXRsw8+yv8E/RCfpLvgbTEpXQR7BTXPSbWubDXPg4bIa3Cha51J2Yt+ZyDKxH+Uy34xmlPYwBn5DdO7v+DFw2QbSUYp2TpPKUkFF34SsQ4c6eUVRapeVqemI0a6DFnYt1AG1drl/7BBeWqdPYSe80bRRGaG1XYNFEXY9Ip2NutwcE2QcVUQVQT24CG8A9c6bajOgVOEAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAXCAYAAACf+8ZRAAADvUlEQVR4Xp1Wy4uOURz+nZqpERqMaGrkkksWMjWkhCwQC9Pk1oS/ACtCyUKhZiw0kQbZzEJEimQhYspGLJTCwkI0mpWNzEaJ5zmX95z3XL7vG0893znndznnOec9l08kgAobjRAGtpw0jbSS09kTf2LImBKDRWJPDBW0J+NOTC3PtIZyZNnj0SRmDiK6pHFYOzgzNmZTShMMVyiTJnG0b9Wi2TgBPgIvgTckEabjZ4DXwP7InjSWgIOJUI9VYga6CftB0R0n8jkYfYxhLHNC9MH+EeUyUKF+Hl18R/0ouBjsAfeD78BRsC0ndTV4BHwBwx+UYwXRe8UM1gvOEg4m8gzs9CHSqYyNPsb0ogPmMNd1dhx8Bc42KbIJPClGxyBiDqE8gNCHYiZQh+5D6eAB1DeinADHvLPCIvAzyA4d5oJvwWO6ZTo7bW30OTDnE7jQttn/uJhJEX3IvWrrBHpSw1LbFg51UUQ3+FWc6DowsJoSDuDBHm6LF+AmUeXbRVmP31/iRZyRQDRi6L9g68R21EZQtplmiOmIVnIlI5p2xk4qsz/xxdQPifMVc3TuRWvZKuar9VgR3JpuQgvAe8bnEB5ob3PoRr0keqwSbeNtwVhn54RYj/NjO6+xUcTfF3P47opZdR44rnB/1XuqNoT2VisdxbLD8exKh6JFdiPmr7bVx/SifcesrRW9FfRtQ1O/Fc1tQf8G8DK4U7dronyjtD14hz4Hp1Rj0btAI1qj6jgRrYtQhNLb4YH4bcHbhmOuAM+Ce6zdoiS63nMoLkRoj7eBQ2KPRHNl7bbQ4BX6WlU3leKtc130Fw9nSlclWsWDEjxEJdETyqwQDqOatLYQ7iDy1oigRVCsvy2U3jY/RW+3KmgIv9SXwB5EZfZ0fVI8HHx4tjkD3B0onlh2wGD3vm37QOb8liA3QLwtGO++jBWthQyJeS1jXdXtwbvXuqqiC79vwHPGoLFczGM0GNgOg9/ApbbNDvg68nOHLyeBlVUjyhzGEJwA+3UrzcXgcz4/VMwV4CfmSuIg6RuAj8F7cE0VJbIO/AKeAvcJHxIlfLnag8m76+ylmFeWgj+IefrjVeK2GFax1WyTW/SJ8W0W8/iYUB0dpsTpKXiT7AAHxDztHj6XtZUoOLEtaHEiMfjlKIxlHaafeeAd8Cn4WFXbxw1iy0Rvk8lkTHkkga0MGKJkF3t3NxH6X8hoTMYqoZWYtOewnvsP0Cqmm1uKj6yJxoYoRZTsFk3cIVr+Gjm0kpePyVgTU2aprNq8aCX/AEIcm+IxOBY/AAAAAElFTkSuQmCC>