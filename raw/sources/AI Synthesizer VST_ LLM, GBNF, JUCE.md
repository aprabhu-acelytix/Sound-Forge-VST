# **Deterministic Architecture for Autonomous Sound Design: A Deep Dive into LLM Prompting, Grammar-Constrained Decoding, and Real-Time DSP State Management**

The integration of generative artificial intelligence with digital signal processing (DSP) presents a uniquely complex systems engineering challenge. Architecting a hybrid VST3 synthesizer plugin—specifically, one where a Large Language Model (LLM) acts as an autonomous "Brain" to translate human semantic intent into precise mathematical audio parameters—requires traversing multiple, often conflicting, technical domains. The overarching system must seamlessly connect high-latency, non-deterministic natural language processing with the hard real-time, microsecond-critical execution environments of modern C++ audio threads.1

This exhaustive research report details the architectural blueprint for "Sound Forge," a theoretical hybrid VST3 synthesizer built upon the JUCE framework. The primary objective is to establish a rigorous methodology for forcing an LLM to generate mathematically reliable, structurally perfect JSON payloads, and subsequently applying those parameters to the DSP engine without inducing thread deadlocks, priority inversion, or audio artifacts such as zipper noise. The analysis is divided into four distinct phases: deterministic prompt engineering, grammar-constrained token decoding, lock-free thread synchronization, and audio-rate parameter smoothing.

## **1\. Deterministic Prompt Engineering for Sound Design**

The primary challenge in utilizing an LLM for synthesizer patch generation lies in the vast dimensionality of the parameter matrix. A modern hybrid synthesizer regularly contains over 100 distinct manipulable parameters—ranging from oscillator waveforms and discrete filter cutoff frequencies to complex modulation matrices and envelope stages.3 The LLM must not only understand the user's semantic request (e.g., "create a warm, evolving cinematic pad") but must also accurately translate that conceptual intent into exact floating-point values that govern the physics of sound synthesis.5 Standard prompting techniques are thoroughly insufficient for this degree of precision; therefore, a highly structured methodology relying on Few-Shot Learning and Chain-of-Thought (CoT) reasoning is absolutely required.7

### **1.1 The Dimensionality Problem and System Prompt Structuring**

To effectively leverage language models for complex parameter generation, the system prompt must serve three critical functions: establishing role definition, strictly defining the task boundaries, and enforcing a rigid output format.3 An LLM left to its own devices will optimize for linguistic plausibility rather than mathematical accuracy.10 By explicitly defining the model as an expert audio engineer and DSP architect, the model's latent weights regarding acoustic physics, frequency spectrums, and synthesizer architecture are brought to the forefront of the context window.11

Furthermore, the system prompt must categorically define the synthesizer's architecture. The LLM cannot guess the available parameters; it must be provided with a strict ontological map of the VST. If the model generates a key that the plugin does not recognize, the data will be silently dropped or cause a parsing failure.

Table 1 illustrates the necessary architectural mapping that must be explicitly embedded within the system prompt to guide the LLM's understanding of the bounded parameter space.

| Parameter Group | Parameter ID | Data Type | Range/Constraints | Acoustic Function |
| :---- | :---- | :---- | :---- | :---- |
| **Oscillator 1** | osc1\_wave | Integer | 0 (Sine) to 3 (Saw) | Defines the fundamental harmonic content and initial timbre. |
| **Oscillator 1** | osc1\_detune | Float | \-1.0 to 1.0 (Semitones) | Induces phase cancellation and chorusing for stereo width. |
| **Filter (VCF)** | vcf\_cutoff | Float | 20.0 to 20000.0 (Hz) | Determines the spectral ceiling of the patch; shapes brightness. |
| **Filter (VCF)** | vcf\_res | Float | 0.1 to 10.0 (Q Factor) | Emphasizes the cutoff frequency, adding resonance or self-oscillation. |
| **Amp Env (VCA)** | vca\_attack | Float | 0.001 to 5.0 (Seconds) | Governs the initial transient strike and volume onset. |
| **Amp Env (VCA)** | vca\_decay | Float | 0.001 to 5.0 (Seconds) | Controls the fall from peak amplitude to the sustain level. |
| **Amp Env (VCA)** | vca\_sustain | Float | 0.0 to 1.0 (Multiplier) | Sets the resting volume level while a MIDI note is held. |
| **Amp Env (VCA)** | vca\_release | Float | 0.001 to 10.0 (Seconds) | Controls the ambient volume tail after MIDI note-off. |

### **1.2 Few-Shot Prompting and In-Context Learning**

Research indicates that Large Language Models operate highly effectively as "few-shot learners".13 Zero-shot prompting—asking the model to perform a task without providing prior examples—frequently results in hallucinated parameter names or values that fall completely outside the acceptable numerical ranges programmed into the VST.11 Few-shot prompting mitigates this instability by providing the model with a series of high-quality demonstrations directly within the prompt context.10

For a synthesizer parameter matrix, these examples act as deterministic contextual templates.15 By providing three to five explicit examples of text-to-parameter mappings (e.g., mapping "aggressive sub bass" to a fast VCA attack, high oscillator saturation, and a 24dB/octave low-pass filter set to 120Hz), the LLM infers the underlying acoustic physics linking language to DSP logic.11 This technique drastically reduces the need for computationally expensive fine-tuning of the model weights, relying instead on the model's in-context recall to establish the correlation between subjective human adjectives and objective floating-point DSP states.8 When the model observes that "punchy" consistently correlates with vca\_attack \< 0.015, it applies this mathematical relationship to future generations.

### **1.3 Chain-of-Thought (CoT) and \<think\> Blocks for Acoustic Reasoning**

While few-shot prompting establishes the output format, Chain-of-Thought (CoT) prompting enables the complex reasoning required to calculate interdependent variables.17 CoT operates on the underlying principle that forcing an LLM to generate explicit intermediate reasoning steps before arriving at a final answer drastically improves zero-shot and few-shot accuracy.13

In the specific context of autonomous sound design, an LLM cannot be expected to simply jump from the word "plucky" to an isolated parameter value of {"vca\_decay": 0.15}. It must "show its work" to avoid statistical hallucinations. Utilizing designated \<think\> blocks forces the model to deliberate on acoustic physics prior to outputting the final, minified JSON payload.20 If the model is prompted to create a "rhythmic, tempo-synced trance gate at 120 BPM," the CoT mechanism forces it to calculate the mathematical relationship between tempo and time: 120 BPM equates to 2 beats per second, or 500 milliseconds per beat. Therefore, a 16th-note modulation requires an LFO rate of exactly 125 milliseconds.

This process mimics the sequential cognitive workflow of a human sound designer.17 By explicitly outlining the intermediate reasoning steps in the text output, the CoT mechanism prevents the model from predicting statistically likely but mathematically incorrect tokens, effectively grounding the LLM in the physical constraints of the audio engine.21

### **1.4 Theoretical Prompt Template for Hybrid Synthesizers**

Combining strict role definition, parameter constraint declarations, few-shot examples, and CoT reasoning yields a robust system prompt framework.9 The following is an exhaustive theoretical template designed to force maximum determinism from the LLM, ensuring that it remains locked into the persona of a DSP architect:

You are an elite DSP architect and expert synthesizer sound designer.

Your objective is to translate user semantic requests into exact parameter values for "Sound Forge," a hybrid VST3 synthesizer.

You must strictly adhere to the following parameter IDs and bounds. NEVER hallucinate a parameter ID that is not on this list, and NEVER output a value outside the specified range:

* vcf\_cutoff: Float \[20.0, 20000.0\]  
* vcf\_res: Float \[0.1, 10.0\]  
* vca\_attack: Float \[0.001, 5.0\]  
* vca\_decay: Float \[0.001, 5.0\]  
* vca\_sustain: Float \[0.0, 1.0\]  
* vca\_release: Float \[0.001, 10.0\]  
* lfo1\_rate: Float \[0.1, 50.0\]  
* lfo1\_depth: Float \[0.0, 1.0\]  
1. You must first output a block where you reason step-by-step about the acoustic physics required to achieve the user's sound.  
2. Analyze the transients, the spectral envelope, the harmonic content, and the necessary modulations.  
3. After the tag, you must output a raw, minified JSON object containing ONLY the exact parameter keys and their corresponding float values.  
4. Do not include markdown formatting likejson around the final output.

User: "Create a punchy, aggressive analog bass pluck."

1. A "pluck" requires an immediate transient. VCA Attack must be near minimum (0.005s).  
2. The sound must decay quickly to create the plucky articulation. VCA Decay should be short (0.2s) with zero Sustain (0.0), and a short Release (0.1s).  
3. "Aggressive analog bass" implies a low cutoff frequency that allows low-end energy but requires some resonance to accentuate the strike. VCF Cutoff set to 150.0Hz with a Resonance of 2.5.  
4. No LFO modulation is required for a standard bass pluck.  
   {"vca\_attack":0.005,"vca\_decay":0.200,"vca\_sustain":0.0,"vca\_release":0.100,"vcf\_cutoff":150.0,"vcf\_res":2.5,"lfo1\_rate":0.1,"lfo1\_depth":0.0}

"{user\_prompt}"

\#\#\# 1.5 LLM Hyperparameter Tuning for DSP Determinism

Prompt engineering alone is insufficient without proper configuration of the LLM's inference hyperparameters. The \`temperature\` and \`top\_p\` (nucleus sampling) settings directly control the entropy of the token selection process.\[23\] For creative writing, higher temperatures encourage diversity. However, for generating executable parameters intended for a C++ backend, creativity is a liability.

To achieve maximum determinism, the model's \`temperature\` must be set extremely low (e.g., \`0.0\` or \`0.1\`).\[23\] This forces the model into a greedy decoding state, wherein it invariably selects the highest-probability token at every step.\[23\] Setting the \`top\_p\` parameter to a restricted value (e.g., \`0.1\`) further truncates the long tail of the probability distribution, ensuring the model does not attempt to sample from unlikely, mathematically incoherent token sequences.\[23\]

\#\# 2\. Grammar-Constrained Decoding (GBNF)

Even with an optimal system prompt, rigorous few-shot examples, and greedy decoding hyperparameters, LLMs fundamentally remain stochastic, probabilistic engines. There is always a non-zero probability that the model will generate an invalid JSON character, a hallucinated parameter key, or a string where a floating-point number is required. In a C++ application interfacing directly with a digital audio workstation (DAW), a single unescaped quote or malformed JSON payload will cause a parsing failure, silently breaking the AI integration or causing the plugin to crash.\[24, 25\] 

To bridge the gap between stochastic text generation and the rigid structural requirements of a VST3 plugin, the system must utilize Grammar-Constrained Decoding. Specifically, the implementation of GGML BNF (GBNF) grammars within the \`llama.cpp\` inference engine represents the most robust solution for physical token restriction.\[26, 27\]

\#\#\# 2.1 The Mechanics of GBNF Token Masking

Traditional "JSON Mode" offered by many commercial API services relies primarily on prompt fine-tuning and basic post-generation output validation.\[27\] However, GBNF operates fundamentally differently by intervening directly at the decoding step during inference. Language models predict the probability distribution (logits) of the next token in a sequence.\[27\] GBNF parses a predefined Backus-Naur Form grammar and evaluates the current state of the generated text against the Abstract Syntax Tree (AST) of the grammar.\[28, 29\]

If the AST dictates that the next character must be a digit (e.g., to fulfill a floating-point requirement for the \`vcf\_cutoff\` parameter), the GBNF engine iterates through the entire LLM vocabulary. Any token that contains a letter, a line break, or an invalid punctuation mark is assigned a probability of absolute zero (or negative infinity in the logit space).\[24, 27\] The model is then mechanically forced to sample only from the remaining valid tokens. This effectively guarantees that the output will strictly adhere to the provided schema, completely eliminating syntax errors, missing brackets, and structural hallucinations.

\#\#\# 2.2 Constructing a Strict Grammar for Synthesizer Parameters

To construct a GBNF file that forces the LLM to output a specific JSON structure of \`{"parameter\_id": float\_value}\`, rules must be established for the root object, whitespace, numbers, strings, and the specific key-value pairs allowed.\[26, 30\] 

The Backus-Naur Form utilizes non-terminal symbols (rule names) and terminal symbols (actual characters or Unicode points).\[29\] A robust grammar for "Sound Forge" must explicitly define the allowed synthesizer parameters as the \*only\* valid strings for the JSON keys.\[31\] By hardcoding the exact parameter IDs into the grammar itself, it becomes physically impossible for the LLM to hallucinate a parameter that does not exist in the JUCE \`AudioProcessorValueTreeState\`.\[28\]

\#\#\# 2.3 Boilerplate GBNF Schema Implementation

The following is an exhaustive boilerplate GBNF schema tailored specifically for a hybrid VST environment requiring exact float mapping. Note that to accommodate the CoT \`\<think\>\` block discussed in Section 1, the root rule must be expanded to allow arbitrary text generation \*before\* the JSON object is enforced.

\`\`\`gbnf  
\# 1\. Define the root structure.   
\# It allows an optional CoT think block, followed by the strict JSON object.  
root ::= ("\<think\>\\n" \[^\<\]\* "\</think\>\\n")? "{" ws kv-pair ("," ws kv-pair)\* "}" ws

\# 2\. Define whitespace (optional spaces, tabs, or newlines)  
\# Utilizing {0,20} limits extreme repetition which can cause severe performance penalties in llama.cpp  
ws ::= \[ \\t\\n\]{0,20}

\# 3\. Define the specifically allowed parameter keys.   
\# This prevents the LLM from inventing parameters. It MUST choose from this exact string list.  
allowed-keys ::= "\\"vcf\_cutoff\\"" | "\\"vcf\_res\\"" | "\\"vca\_attack\\"" | "\\"vca\_decay\\"" | "\\"vca\_sustain\\"" | "\\"vca\_release\\"" | "\\"lfo1\_rate\\"" | "\\"lfo1\_depth\\""

\# 4\. Define the floating point number structure.  
\# Accounts for optional negative signs, integral parts, required decimal points, and fractional parts.  
\# Constraining the length of the fractional part prevents infinite generation loops.  
integral-part ::= | \[1-9\]\[0-9\]{0,5}  
fractional-part ::= \[0-9\]{1,6}  
number ::= ("-"? integral-part) "." fractional-part

\# 5\. Define the key-value pair.  
\# Enforces the JSON dictionary structure: "key" : 0.00  
kv-pair ::= allowed-keys ws ":" ws number

### **2.4 Performance Considerations and Edge Cases**

While GBNF is exceptionally powerful for enforcing structured outputs, it introduces notable performance considerations during inference. Excessive use of optional repetitions (e.g., using x? x? x? instead of bounded ranges like x{0,N}) can result in exponential branching within the grammar parser's state machine, leading to extreme sampling slowdowns and CPU spikes.30 The usage of bounded repetitions, such as \[0-9\]{1,6} for the fractional part of the float, ensures the AST evaluation remains highly performant.28

Furthermore, it is critical to note that while GBNF guarantees structural validity up to the point of generation, it does not guarantee that the LLM will successfully complete the generation before hitting the context or token limits.24 Truncated JSON is a persistent risk if the max\_tokens parameter is set too low or if the preceding \<think\> block is overly verbose. The background AI thread in JUCE must still wrap the JSON parser in a robust try-catch block to gracefully handle unexpectedly truncated strings and avoid undefined behavior.24

Additionally, when utilizing JSON schema conversion tools (like llama.cpp's json\_schema\_to\_grammar.py), it is advised to set additionalProperties to false.32 Allowing additional properties requires the grammar to support arbitrary string generation for keys, which re-introduces the risk of parameter hallucination and slows down token sampling significantly due to the expanded state space.31

## **3\. Thread-Safe APVTS Integration (The C++ Architecture)**

Once the autonomous LLM has generated a verified, mathematically sound JSON payload, the data must be safely transferred from the background AI inference thread into the digital signal processing architecture of the plugin. In the JUCE framework, plugin state is overwhelmingly managed by the AudioProcessorValueTreeState (APVTS), which bridges the UI, the host DAW automation, and the audio processor.33

However, interacting with the APVTS and the audio thread requires navigating a minefield of potential multithreading disasters. Implementing this pipeline incorrectly will lead to immediate host DAW crashes, deadlocks, and severe priority inversion.34

### **3.1 The Dangers of APVTS Contention and Priority Inversion**

The audio callback (processBlock) provided by JUCE runs on a highly specialized, ultra-high-priority thread managed directly by the host DAW (e.g., Ableton Live, Logic Pro, Pro Tools).1 This thread has strict real-time deadlines; it must compute and fill the audio buffer in a matter of milliseconds or microseconds. If the audio thread is forced to wait for an operating system lock, it will miss its deadline, resulting in catastrophic audio dropouts, buffer underruns, and audible clicks.35

The AudioProcessorValueTreeState relies heavily on juce::ValueTree, a class that is fundamentally not thread-safe for concurrent modification.34 When a parameter is updated via the standard setValueNotifyingHost() function, the APVTS iterates through a ListenerList to notify the GUI components and the host DAW of the change.35 This iteration is rigidly protected by a ScopedLock (a mutex) to prevent listeners from being added or removed while the list is being traversed, which would otherwise result in out-of-bounds memory access and hard crashes.35

If an asynchronous background worker thread (such as the low-priority thread executing the LLM inference) calls APVTS::getParameter()-\>setValueNotifyingHost() directly, it acquires that mutex.35 If the host DAW or the GUI thread simultaneously attempts to read or modify a parameter, or if a listener attempts to register/deregister (such as when the user opens or closes the plugin UI), they will violently contend for the same lock.35

More critically, this scenario creates **priority inversion**. If a low-priority AI thread acquires the listener lock to push a parameter update, and the ultra-high-priority audio thread subsequently attempts to access related parameter data or triggers a callback that touches the same lock space, the high-priority audio thread is forced to sleep until the low-priority AI thread finishes its execution and releases the lock.36 The OS scheduler may even preempt the AI thread while it holds the lock, effectively freezing the entire audio engine indefinitely.38

For this reason, the absolute cardinal rule of JUCE multithreading is: **Never update APVTS parameters or call setValueNotifyingHost() directly from an audio thread or a background worker thread.** All APVTS modifications must occur strictly on the JUCE Message Thread (the main GUI/event thread).37

### **3.2 The SPSC Lock-Free FIFO Pipeline**

To safely transfer the generated parameters from the AI thread to the Message Thread without incurring locks, the architecture must utilize a lock-free Single-Producer, Single-Consumer (SPSC) queue.40

Lock-free queues utilize std::atomic variables and strict memory ordering (std::memory\_order\_acquire and std::memory\_order\_release) to allow two distinct threads to share data without ever invoking an operating system mutex.41 The background AI thread acts as the sole producer, pushing a parsed struct of parameter values into the FIFO buffer. The Message Thread acts as the sole consumer, pulling data out of the FIFO. Because neither thread blocks or locks the other, priority inversion is mathematically impossible.41 In JUCE, this is typically implemented using the juce::AbstractFifo class paired with a pre-allocated circular buffer (e.g., std::array or std::vector), entirely avoiding memory allocation during the transfer process.40

### **3.3 Message Thread Synchronization: juce::AsyncUpdater vs juce::Timer**

Once the parameter data resides safely in the lock-free FIFO, the Message Thread must be notified to read it. There are two primary mechanisms in JUCE to facilitate this transition: juce::AsyncUpdater and juce::Timer.2

The juce::AsyncUpdater provides a triggerAsyncUpdate() method that can be called directly from the background AI thread. This mechanism posts a message to the operating system's native message loop, coalescing multiple calls, which eventually invokes handleAsyncUpdate() on the JUCE Message Thread.2 While highly convenient, triggerAsyncUpdate() is not strictly real-time safe.43 On certain operating systems (particularly Windows), posting to the message queue via SendMessage can trigger hidden memory allocations or briefly block the calling thread while it interacts with internal OS structures.44 If the AI thread fires updates too rapidly, it can flood the OS message queue, leading to UI stutter.46

A generally safer and highly performant alternative for audio parameter updates is the juce::Timer class.43 By instantiating a Timer inside a UI component or a dedicated background manager that runs inherently on the Message Thread, the plugin can poll the lock-free FIFO at a fixed, reliable rate (e.g., 60 Hz). Polling an empty lock-free queue is exceptionally cheap—costing little more than a single atomic read operation—and entirely avoids interacting with the unpredictable OS message allocator.44

### **3.4 C++ Boilerplate: The AI-to-DSP Synchronization Pattern**

The following code illustrates the proper architectural pipeline using a lock-free juce::AbstractFifo, a pre-allocated std::array, and a juce::Timer to safely bridge the AI inference thread and the APVTS without risking priority inversion.

C++

\#**include** \<JuceHeader.h\>  
\#**include** \<array\>  
\#**include** \<atomic\>

// 1\. Define a Plain Old Data (POD) struct to hold the parsed JSON payload  
struct AIPayload {  
    juce::String parameterID;  
    float newValue;  
};

// 2\. Implement the SPSC Lock-Free Queue using AbstractFifo  
class AIFifoQueue {  
public:  
    AIFifoQueue() : fifo (1024) {}

    bool push (const AIPayload& payload) {  
        // AbstractFifo manages the read/write indices atomically  
        auto writeHandle \= fifo.write (1);  
        if (writeHandle.blockSize1 \> 0) {  
            buffer\[(size\_t) writeHandle.startIndex1\] \= payload;  
            return true;  
        }  
        return false; // Queue is full, drop or handle gracefully  
    }

    bool pop (AIPayload& payload) {  
        auto readHandle \= fifo.read (1);  
        if (readHandle.blockSize1 \> 0) {  
            payload \= buffer\[(size\_t) readHandle.startIndex1\];  
            return true;  
        }  
        return false; // Queue is empty, nothing to do  
    }

private:  
    juce::AbstractFifo fifo;  
    std::array\<AIPayload, 1024\> buffer; // Pre-allocated memory, no mallocs on audio/AI threads  
};

// 3\. The Background AI Thread (The Single Producer)  
class AIWorkerThread : public juce::Thread {  
public:  
    AIWorkerThread (AIFifoQueue& queueToUse)   
        : juce::Thread ("LLM\_Inference\_Thread"), queue (queueToUse) {}

    void run() override {  
        //... LLM Inference & GBNF Generation occurs here...  
        //... JSON parsing extracts the key/value pairs...  
          
        AIPayload newParam;  
        newParam.parameterID \= "vcf\_cutoff";  
        newParam.newValue \= 1200.5f;

        // Push data to the lock-free queue (No OS locks, 100% thread safe)  
        queue.push (newParam);  
    }  
private:  
    AIFifoQueue& queue;  
};

// 4\. The Message Thread Poller (The Single Consumer & APVTS Updater)  
class AIParameterDispatcher : public juce::Timer {  
public:  
    AIParameterDispatcher (AIFifoQueue& queueToUse, juce::AudioProcessorValueTreeState& apvtsToUse)  
        : queue(queueToUse), apvts(apvtsToUse) {  
        startTimerHz (60); // Poll at 60 FPS, tied to the Message Thread  
    }

    void timerCallback() override {  
        AIPayload payload;  
          
        // Drain the FIFO until empty  
        while (queue.pop (payload)) {  
            // Retrieve the parameter object from the APVTS  
            if (auto\* param \= apvts.getParameter (payload.parameterID)) {  
                // Safely update the APVTS and notify the host DAW  
                // Executing this here guarantees it runs on the Message Thread,  
                // safely acquiring the ListenerList ScopedLock without contesting the audio thread.  
                param-\>beginChangeGesture();  
                param-\>setValueNotifyingHost (param-\>convertTo0to1 (payload.newValue));  
                param-\>endChangeGesture();  
            }  
        }  
    }  
private:  
    AIFifoQueue& queue;  
    juce::AudioProcessorValueTreeState& apvts;  
};

This architecture completely decouples the non-deterministic, long-running execution time of the LLM from the highly sensitive, low-latency internal workings of the host DAW, ensuring robust plugin stability across all major platforms.

## **4\. Parameter Smoothing & Audio Artifact Prevention**

The final hurdle in implementing an autonomous AI sound designer lies within the core DSP audio engine itself. When the AIParameterDispatcher successfully updates the APVTS on the Message Thread, the underlying std::atomic\<float\> associated with that parameter instantaneously changes its value in memory.49

The high-priority audio thread, iteratively executing the processBlock at a rate of 44,100 to 96,000 times per second, continuously reads this atomic value. If an AI-generated parameter dictates that the filter cutoff frequency should jump from 400Hz to 8,000Hz, the audio engine will process sample ![][image1] at 400Hz and sample ![][image2] at 8,000Hz.

### **4.1 The Physics of Zipper Noise and Discontinuities**

In digital signal processing, an instantaneous jump in amplitude, frequency, or phase creates a severe discontinuity in the waveform.50 According to Fourier theory, an abrupt vertical step function requires infinite high-frequency harmonics to represent mathematically. In the acoustic domain, this translates directly to a broadband transient click, pop, or harsh "zipper noise".51 If the LLM generates a rapid sequence of parameter changes—such as evolving an envelope or modulating a wavetable index—the resulting continuous discontinuities will entirely ruin the audio signal, rendering the synthesizer unusable.52

Furthermore, updating complex DSP objects—such as calculating new coefficients for Infinite Impulse Response (IIR) filters—is computationally heavy. If coefficients are recalculated on every single sample without interpolation, the CPU load will spike drastically.52 Conversely, if coefficients are only updated once per audio block (e.g., every 512 samples), the stair-stepping effect exacerbates the zipper noise.52 Therefore, parameter jumps must be interpolated at audio rates using sub-block or per-sample smoothing.

### **4.2 Parameter Interpolation via juce::SmoothedValue**

The JUCE framework provides the juce::SmoothedValue template class to handle per-sample parameter interpolation.55 Rather than abruptly switching states, SmoothedValue applies a low-pass filter (a ramp) to the control signal, smoothly gliding from the old parameter state to the newly generated AI parameter state over a specified number of milliseconds or samples.

There are two critical acoustic considerations when configuring parameter smoothing for a synthesizer:

1. **Ramp Length:** A ramp length that is too short (e.g., 1ms) will fail to eliminate zipper noise during drastic AI-driven parameter jumps. Conversely, a ramp length that is too long (e.g., 500ms) will introduce extreme lag, destroying the immediate transient response required for plucks, drums, and percussive patches.51 A balanced ramp time of 10ms to 25ms is generally ideal for synthesizer control rates, providing a smooth transition without noticeable auditory delay.57  
2. **Smoothing Type (Linear vs. Multiplicative):** Linear smoothing (juce::ValueSmoothingTypes::Linear) is appropriate for parameters like stereo panning, dry/wet mix, or linear LFO depth.58 However, human hearing perceives pitch (Hz) and loudness (dB) logarithmically. If a linear smoother is applied to a VCF cutoff frequency jumping from 20Hz to 20,000Hz, the sweep will sound highly unnatural, spending far too much acoustic time in the upper high frequencies. For frequencies, delay times, and raw gain multipliers, juce::ValueSmoothingTypes::Multiplicative must be explicitly utilized.56 This applies an exponential smoothing curve, ensuring equal power crossfades and musically pleasing, pitch-accurate frequency sweeps.51

### **4.3 Implementing Smoothing Inside the DSP Loop**

The implementation of SmoothedValue requires a highly specific sequence of operations within the audio processor to ensure thread safety and DSP efficiency. The smoother must be initialized in prepareToPlay() with the current sample rate, its target must be updated once per block via setTargetValue(), and it must be polled once per sample via getNextValue().61

The following C++ architecture demonstrates the integration of the AI-driven APVTS atomic pointers with multiplicative parameter smoothing inside the critical audio path. It addresses the common pitfall of start-up clicks by ensuring the smoother initializes to the current parameter value rather than ramping up from zero.61

C++

class SoundForgeProcessor : public juce::AudioProcessor {  
public:  
    // Pointer to the lock-free atomic value managed internally by the APVTS  
    std::atomic\<float\>\* vcfCutoffRaw \= nullptr;  
      
    // The multiplicative smoother for logarithmic frequency values  
    juce::SmoothedValue\<float, juce::ValueSmoothingTypes::Multiplicative\> vcfCutoffSmoothed;  
      
    // Standard IIR Low Pass Filter  
    juce::dsp::IIR::Filter\<float\> lowPassFilter;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override {  
        // Initialize the smoother with the current sample rate and a 15ms ramp time  
        vcfCutoffSmoothed.reset (sampleRate, 0.015);  
          
        // Prevent the smoother from starting at 0.0 and ramping up on playback start.  
        // It must jump instantly to the initial saved state to prevent start-up clicks.  
        if (vcfCutoffRaw\!= nullptr) {  
            vcfCutoffSmoothed.setCurrentAndTargetValue (vcfCutoffRaw-\>load());  
        }  
          
        juce::dsp::ProcessSpec spec { sampleRate, static\_cast\<juce::uint32\> (samplesPerBlock), 2 };  
        lowPassFilter.prepare (spec);  
    }

    void processBlock (juce::AudioBuffer\<float\>& buffer, juce::MidiBuffer& midiMessages) override {  
        juce::ScopedNoDenormals noDenormals;

        // 1\. Read the atomic float safely at the beginning of the audio block.  
        // This value was set asynchronously by the AI Message Thread Poller.  
        float currentTargetHz \= vcfCutoffRaw-\>load (std::memory\_order\_relaxed);  
          
        // 2\. Feed the new AI value into the smoother.   
        // If the value hasn't changed, this operation is highly optimized and cheap.  
        vcfCutoffSmoothed.setTargetValue (currentTargetHz);

        auto\* leftChannel  \= buffer.getWritePointer (0);  
        auto\* rightChannel \= buffer.getWritePointer (1);

        // 3\. Process the audio sample by sample  
        for (int sample \= 0; sample \< buffer.getNumSamples(); \++sample) {  
              
            // 4\. Calculate the exponentially interpolated parameter value for this exact microsecond  
            float interpolatedCutoff \= vcfCutoffSmoothed.getNextValue();  
              
            // 5\. Update the filter coefficients based on the smoothed value.  
            // (Note: For extreme performance in production environments, StateVariableFilters   
            // handle coefficient updates much more efficiently per-sample than standard IIR filters)  
            updateFilterCoefficients (interpolatedCutoff);  
              
            // 6\. Apply the Digital Signal Processing  
            float inputL \= leftChannel\[sample\];  
            float inputR \= rightChannel\[sample\];  
              
            leftChannel\[sample\]  \= lowPassFilter.processSample (inputL);  
            rightChannel\[sample\] \= lowPassFilter.processSample (inputR);  
        }  
    }  
      
private:  
    void updateFilterCoefficients(float cutoffHz) {  
        // Logic to recalculate IIR coefficients based on cutoffHz and sample rate  
        //...  
    }  
};

This pattern guarantees that regardless of how radically or chaotically the LLM alters the parameters of the synthesizer, the DSP engine will process the transitions fluidly.62 The multiplicative ramp absorbs the shock of instantaneous atomic memory updates, ensuring the resulting audio output remains entirely free of aliasing, zipper noise, and digital artifacts, resulting in a professional, artifact-free synthesis engine.52

## **Conclusion**

Architecting a hybrid VST3 synthesizer capable of translating non-deterministic LLM natural language into strict DSP physics requires a multi-layered, highly defensive engineering strategy. The system must first rely on heavily constrained Chain-of-Thought prompting and Few-Shot learning to map abstract semantic human intent to logical parameter matrices. This is followed immediately by GGML BNF grammar decoding to physically restrict the LLM's token generation probabilities to the exact, required JSON structure, eliminating the possibility of hallucinated syntax.

Once a mathematically sound payload is generated by the AI, strict C++ systems engineering principles dictate the usage of a Single-Producer, Single-Consumer lock-free FIFO queue. This queue operates in tandem with a juce::Timer on the Message Thread to completely decouple the AI's execution latency from the real-time audio thread, strictly preventing APVTS ScopedLock contention and the fatal system failure of priority inversion. Finally, the raw parameter outputs are passed through exponential juce::SmoothedValue ramps within the inner loop of the processBlock.

By strictly adhering to this comprehensive pipeline—from linguistic prompt structuring and token masking down to sub-sample coefficient interpolation—the unpredictable nature of a Large Language Model is securely bridged to the unforgiving, hard real-time requirements of commercial audio software.

#### **Works cited**

1. Need some understanding on the message thread and async calls \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/need-some-understanding-on-the-message-thread-and-async-calls/61573](https://forum.juce.com/t/need-some-understanding-on-the-message-thread-and-async-calls/61573)  
2. juce::AsyncUpdater Class Reference, accessed May 1, 2026, [https://docs.juce.com/master/classjuce\_1\_1AsyncUpdater.html](https://docs.juce.com/master/classjuce_1_1AsyncUpdater.html)  
3. Can Large Language Models Predict Audio Effects Parameters from Natural Language?, accessed May 1, 2026, [https://arxiv.org/html/2505.20770v2](https://arxiv.org/html/2505.20770v2)  
4. Creative Text-to-Audio Generation via Synthesizer Programming \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2406.00294v1](https://arxiv.org/html/2406.00294v1)  
5. Unleashing the potential of prompt engineering for large language models \- PMC, accessed May 1, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12191768/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12191768/)  
6. An Introduction to Large Language Models: Prompt Engineering and P-Tuning | NVIDIA Technical Blog, accessed May 1, 2026, [https://developer.nvidia.com/blog/an-introduction-to-large-language-models-prompt-engineering-and-p-tuning/](https://developer.nvidia.com/blog/an-introduction-to-large-language-models-prompt-engineering-and-p-tuning/)  
7. Few-Shot Prompting \- Prompt Engineering Guide, accessed May 1, 2026, [https://www.promptingguide.ai/techniques/fewshot](https://www.promptingguide.ai/techniques/fewshot)  
8. Chain-of-Thought (CoT) Prompting \- Prompt Engineering Guide, accessed May 1, 2026, [https://www.promptingguide.ai/techniques/cot](https://www.promptingguide.ai/techniques/cot)  
9. LLM Prompt Engineering in Practice: CoT, Few-Shot, and System Prompt Design, accessed May 1, 2026, [https://dev.to/kanta13jp1/llm-prompt-engineering-in-practice-cot-few-shot-and-system-prompt-design-26hf](https://dev.to/kanta13jp1/llm-prompt-engineering-in-practice-cot-few-shot-and-system-prompt-design-26hf)  
10. What is few shot prompting? \- IBM, accessed May 1, 2026, [https://www.ibm.com/think/topics/few-shot-prompting](https://www.ibm.com/think/topics/few-shot-prompting)  
11. Prompt Engineering Techniques for LLMs: A Comprehensive Guide | by Aloy Banerjee, accessed May 1, 2026, [https://medium.com/@aloy.banerjee30/prompt-engineering-techniques-for-llms-a-comprehensive-guide-46ca6466a41f](https://medium.com/@aloy.banerjee30/prompt-engineering-techniques-for-llms-a-comprehensive-guide-46ca6466a41f)  
12. Creative Text-to-Audio Generation via Synthesizer Programming \- MIT Media Lab, accessed May 1, 2026, [https://www.media.mit.edu/projects/ctag/overview/](https://www.media.mit.edu/projects/ctag/overview/)  
13. CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2501.01668v1](https://arxiv.org/html/2501.01668v1)  
14. Prompt Engineering | Lil'Log, accessed May 1, 2026, [https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)  
15. Promptware Engineering: Software Engineering for LLM Prompt Development \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2503.02400v1](https://arxiv.org/html/2503.02400v1)  
16. Few-shot prompt engineering and fine-tuning for LLMs in Amazon Bedrock \- AWS, accessed May 1, 2026, [https://aws.amazon.com/blogs/machine-learning/few-shot-prompt-engineering-and-fine-tuning-for-llms-in-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/few-shot-prompt-engineering-and-fine-tuning-for-llms-in-amazon-bedrock/)  
17. What is chain of thought (CoT) prompting? \- IBM, accessed May 1, 2026, [https://www.ibm.com/think/topics/chain-of-thoughts](https://www.ibm.com/think/topics/chain-of-thoughts)  
18. Chain of Thought Prompting Guide \- PromptHub, accessed May 1, 2026, [https://www.prompthub.us/blog/chain-of-thought-prompting-guide](https://www.prompthub.us/blog/chain-of-thought-prompting-guide)  
19. How Chain of Thought (CoT) Prompting Helps LLMs Reason More Like Humans | Splunk, accessed May 1, 2026, [https://www.splunk.com/en\_us/blog/learn/chain-of-thought-cot-prompting.html](https://www.splunk.com/en_us/blog/learn/chain-of-thought-cot-prompting.html)  
20. Everyone share their favorite chain of thought prompts\! : r/LocalLLaMA \- Reddit, accessed May 1, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1hf7jd2/everyone\_share\_their\_favorite\_chain\_of\_thought/](https://www.reddit.com/r/LocalLLaMA/comments/1hf7jd2/everyone_share_their_favorite_chain_of_thought/)  
21. Towards Physically Plausible Video Generation via VLM Planning \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2503.23368v2](https://arxiv.org/html/2503.23368v2)  
22. CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis \- ACL Anthology, accessed May 1, 2026, [https://aclanthology.org/2025.acl-long.315.pdf](https://aclanthology.org/2025.acl-long.315.pdf)  
23. Using llama-cpp-python grammars to generate JSON \- Simon Willison: TIL, accessed May 1, 2026, [https://til.simonwillison.net/llms/llama-cpp-python-grammars](https://til.simonwillison.net/llms/llama-cpp-python-grammars)  
24. llama.cpp/grammars/README.md at master · ggml-org/llama.cpp ..., accessed May 1, 2026, [https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)  
25. llama.cpp/grammars/README.md at master · ggml-org/llama.cpp · GitHub, accessed May 1, 2026, [https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)  
26. Using Grammar | node-llama-cpp, accessed May 1, 2026, [https://node-llama-cpp.withcat.ai/guide/grammar](https://node-llama-cpp.withcat.ai/guide/grammar)  
27. llama.cpp/grammars/README.md · Steven10429/apply\_lora\_and\_quantize at main, accessed May 1, 2026, [https://huggingface.co/spaces/Steven10429/apply\_lora\_and\_quantize/blame/main/llama.cpp/grammars/README.md](https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/blame/main/llama.cpp/grammars/README.md)  
28. Tutorial: Saving and loading your plug-in state \- JUCE, accessed May 1, 2026, [https://juce.com/tutorials/tutorial\_audio\_processor\_value\_tree\_state/](https://juce.com/tutorials/tutorial_audio_processor_value_tree_state/)  
29. Juce::Value and thread-safety \- General JUCE discussion, accessed May 1, 2026, [https://forum.juce.com/t/juce-value-and-thread-safety/13767](https://forum.juce.com/t/juce-value-and-thread-safety/13767)  
30. Understanding Lock in Audio Thread \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/understanding-lock-in-audio-thread/60007](https://forum.juce.com/t/understanding-lock-in-audio-thread/60007)  
31. Fixed minor race condition and priority inversion \- Development \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/fixed-minor-race-condition-and-priority-inversion/36906](https://forum.juce.com/t/fixed-minor-race-condition-and-priority-inversion/36906)  
32. Thread-Safety of ListenerList \- General JUCE discussion, accessed May 1, 2026, [https://forum.juce.com/t/thread-safety-of-listenerlist/39818](https://forum.juce.com/t/thread-safety-of-listenerlist/39818)  
33. AudioThreadGuard \- keep your audio thread clean \- Useful Tools and Components \- JUCE, accessed May 1, 2026, [https://forum.juce.com/t/audiothreadguard-keep-your-audio-thread-clean/28532](https://forum.juce.com/t/audiothreadguard-keep-your-audio-thread-clean/28532)  
34. VST3 wrapper calls parameterChanged() from audio thread \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/vst3-wrapper-calls-parameterchanged-from-audio-thread/51589](https://forum.juce.com/t/vst3-wrapper-calls-parameterchanged-from-audio-thread/51589)  
35. Getting specific callbacks for each parameter change? \- Page 2 \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/getting-specific-callbacks-for-each-parameter-change/44782?page=2](https://forum.juce.com/t/getting-specific-callbacks-for-each-parameter-change/44782?page=2)  
36. joz-k/LockFreeSpscQueue: A high-performance, single-producer, single-consumer (SPSC) queue implemented in modern C++23 \- GitHub, accessed May 1, 2026, [https://github.com/joz-k/LockFreeSpscQueue](https://github.com/joz-k/LockFreeSpscQueue)  
37. Building a Lock-Free Single Producer, Single Consumer Queue (FIFO) \- Peter Mbanugo, accessed May 1, 2026, [https://pmbanugo.me/blog/building-lock-free-spsc-queue](https://pmbanugo.me/blog/building-lock-free-spsc-queue)  
38. Is MessageManager::callAsync() real-time safe? \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/is-messagemanager-callasync-real-time-safe/66031](https://forum.juce.com/t/is-messagemanager-callasync-real-time-safe/66031)  
39. Lock-free messaging? \- General JUCE discussion, accessed May 1, 2026, [https://forum.juce.com/t/lock-free-messaging/31853](https://forum.juce.com/t/lock-free-messaging/31853)  
40. What's best practice for GUI change notification? \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/whats-best-practice-for-gui-change-notification/12264](https://forum.juce.com/t/whats-best-practice-for-gui-change-notification/12264)  
41. Updating GUI from other threads \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/updating-gui-from-other-threads/20906](https://forum.juce.com/t/updating-gui-from-other-threads/20906)  
42. The big list of JUCE tips and tricks (from n00b to pro) · Melatonin \- Sine Machine, accessed May 1, 2026, [https://melatonin.dev/blog/big-list-of-juce-tips-and-tricks/](https://melatonin.dev/blog/big-list-of-juce-tips-and-tricks/)  
43. Async signalling from Audio Processor to FIFO reader (Editor) (Probably a question for TheVinn and co) \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/async-signalling-from-audio-processor-to-fifo-reader-editor-probably-a-question-for-thevinn-and-co/15228](https://forum.juce.com/t/async-signalling-from-audio-processor-to-fifo-reader-editor-probably-a-question-for-thevinn-and-co/15228)  
44. Reading/writing values lock free to/from processBlock \- Getting Started \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947](https://forum.juce.com/t/reading-writing-values-lock-free-to-from-processblock/50947)  
45. Preventing Audio Artifacts \- General JUCE discussion, accessed May 1, 2026, [https://forum.juce.com/t/preventing-audio-artifacts/49468](https://forum.juce.com/t/preventing-audio-artifacts/49468)  
46. Decreasing lag time on zipper noise smoothing? \- Development \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/decreasing-lag-time-on-zipper-noise-smoothing/26047](https://forum.juce.com/t/decreasing-lag-time-on-zipper-noise-smoothing/26047)  
47. dsp::IIR::Filter: noise/glitches when automating frequency via (slow) LFO \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/dsp-noise-glitches-when-automating-frequency-via-slow-lfo/63949](https://forum.juce.com/t/dsp-noise-glitches-when-automating-frequency-via-slow-lfo/63949)  
48. SmoothedValue, IIRFilter and ProcessorChain \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/smoothedvalue-iirfilter-and-processorchain/42815](https://forum.juce.com/t/smoothedvalue-iirfilter-and-processorchain/42815)  
49. Smoothing Artifacts in Blockwise Processing For Delay \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/smoothing-artifacts-in-blockwise-processing-for-delay/38674](https://forum.juce.com/t/smoothing-artifacts-in-blockwise-processing-for-delay/38674)  
50. How to implement SmoothedValue with DSP Compressor \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/how-to-implement-smoothedvalue-with-dsp-compressor/48189](https://forum.juce.com/t/how-to-implement-smoothedvalue-with-dsp-compressor/48189)  
51. Still glitchy/zipping noises after using smoothedvalue \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/still-glitchy-zipping-noises-after-using-smoothedvalue/59589](https://forum.juce.com/t/still-glitchy-zipping-noises-after-using-smoothedvalue/59589)  
52. Opinion needed on smoothing a Frequency slider which controls HP filter cutoff \- can you improve on my approach here? \- General JUCE discussion, accessed May 1, 2026, [https://forum.juce.com/t/opinion-needed-on-smoothing-a-frequency-slider-which-controls-hp-filter-cutoff-can-you-improve-on-my-approach-here/60264](https://forum.juce.com/t/opinion-needed-on-smoothing-a-frequency-slider-which-controls-hp-filter-cutoff-can-you-improve-on-my-approach-here/60264)  
53. Reducing Zipper Noise in Volume and Pan Changes \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/reducing-zipper-noise-in-volume-and-pan-changes/4372](https://forum.juce.com/t/reducing-zipper-noise-in-volume-and-pan-changes/4372)  
54. Smoothing IIR Filter Response \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/smoothing-iir-filter-response/39347](https://forum.juce.com/t/smoothing-iir-filter-response/39347)  
55. juce::SmoothedValue\< FloatType, SmoothingType \> Class Template Reference, accessed May 1, 2026, [https://docs.juce.com/master/classjuce\_1\_1SmoothedValue.html](https://docs.juce.com/master/classjuce_1_1SmoothedValue.html)  
56. SmoothedValue: should you call getNextValue() for the first sample? \- Audio Plugins \- JUCE, accessed May 1, 2026, [https://forum.juce.com/t/smoothedvalue-should-you-call-getnextvalue-for-the-first-sample/36785](https://forum.juce.com/t/smoothedvalue-should-you-call-getnextvalue-for-the-first-sample/36785)  
57. How to Make Your First VST Plugin | \#07 Smooth Parameter Changes \- YouTube, accessed May 1, 2026, [https://www.youtube.com/watch?v=jn-dhFBrwus](https://www.youtube.com/watch?v=jn-dhFBrwus)  
58. Delay Line artifacts \- Audio Plugins \- JUCE Forum, accessed May 1, 2026, [https://forum.juce.com/t/delay-line-artifacts/46781](https://forum.juce.com/t/delay-line-artifacts/46781)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAABOUlEQVR4XnWRIU9DQRCEb0NJIGmpgIQgEOCqAYchgQREDR4JeEyTYoriN/AHwOAQGBwGi8eQYCsQYDB897i9t9dbJpndudnbvX1tCBGSQ9bWav0CpeM3iONVuh6d4U+1Wl+ohlQ3vYvlh1YrqdFeSg1FdVYrqgYHOjnlJeIRej11zcNt9JC8qj36Wpd8Q77i+IE+ofCAPsO7hJ9w3z5/CM+5tIX3hX5C95uKhDXiOxzZhlM4gMfwB+6Z78GXKfqivR4hoUd8hvewkxo6hDv0C17fvhARX5iGctJGiN8UZEJehNdwWYu6zq4aYMjUbxp2GH6AHtufdUJ4lTyhWWmT8MbhEX0b4lp/pSYsELrWSIn/Q1ZQc00t1x385+skDdltT0kVd8o1TMGBX6+MBLGlcqVKz2xjHOM7+AXJWxziU/9OngAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAYCAYAAABurXSEAAACpklEQVR4XrWXO4hVMRCGJ6ig4AtXFEEL7awsXDsbQdEFF2EbtxFsfNQ2gjZrZaGVVgqCt1ELwcLCxs5m2y0WLGwEW5EFFURE/zxuMsnM5JzL4gf/Tc4/k2RObs65XCID1xr/EWst4VeGiNqI1GAIV/GtvmJZ/U2hTKRYAj1HdzNmWLkzM5cxa85wfr14aMpHg+IpVoeBbBEWhgqydqC5it7eNlAjjHGM2dHi1xlN/j7oMvQc+obgZ7SHqoyCvZRYLDS8r3ICehh6KZ8P6+CLvgTNI/UV2lD08LAGfTFhtJyEHrdmhG+EQglOKBWdHRbcDS1AR9L1NmgeWoQOTpMy4i5Sv66kUzTHvgGXi3bieOyEnkD3oC/QFegtBlxHexcDNtCeNWcOKEE3tmhOPU8pOu00C1+AblDcmR8IvEe7J8V8sh90O13PQilauScdce7Noq9Bx6EleL8ROFNCwf8K3WJei59qjuLEXOcRelauw1c81fY0tkO4gVx0qbfegUe4XEe7nwWWoV/Q6ZwlOQA9gJ42egN9FL4L7cLIza93umEX9AF6DW1Nnm9f4gZWqRyX/jddBztn2nr4uBv6E0rv6XBVPgLlGBTvKIUH062Q/2VydB/tnL6YiijaGiv8YkxIHA/K8SXy55kdAxdfdz+hU9A56E7eoTRILFYjii7YI3nExaL9xh1mdsLRCj7XKD5QU45Bn6B30AtiR6SiWYUZnaL5MHFU/DPij+p36G+U+0PxdXyTj8TT7Pz7usX/yODBpC3xUtkhUXSmW/SsKCuPh+/QAP6Jv8iN+osYpuTbO2Oip9kTify0sPATls+pNmvMgGGsWRRfsTZBms3ewAYlmHZUZ8zESg3JEJaJNUdC1Gj1FaMb5hOLRNI9FTEpR/mj0GXWnDr/H+GeTDgWrmGAAAAAAElFTkSuQmCC>