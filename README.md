# Sionna Neural Receiver: Hybrid 6G AI-RAN Prototype

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch Native](https://img.shields.io/badge/PyTorch-Native-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![NVIDIA 6G Ecosystem](https://img.shields.io/badge/NVIDIA-6G%20Ecosystem-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/industries/telecommunications/ai-ran/)

An implementation of a **6G AI-Native Physical Layer Receiver** utilizing a hybrid cloud-to-edge hardware engineering loop. This repository demonstrates how to bypass local CUDA hardware constraints by shifting heavy 3GPP electromagnetic simulation physics onto cloud GPU nodes via **NVIDIA Sionna 2.0**, while executing model validation, profiling, and inference locally on modern unified memory configurations.

```text
Legacy 5G Loop:  [IQ Tensors] --> [Est. Channel] --> [Equalizer Math] --> [Demapper] --> [Bits]
6G AI-RAN Loop:  [IQ Tensors] ---------------------> [ Neural Receiver ] --------------> [Bits]
```

---

## 🏗️ Architectural Concept: The AI-RAN Shift

Traditional 5G architectures deploy separate, mathematically handcrafted signal processing chains for **Channel Estimation, Equalization, and Bit Demapping**. This project prototypes an **AI-Native 6G Layer 1 Component**—replacing that legacy processing cascade with a single, end-to-end Deep Neural Network (DNN).

### Physics Obstacle & Equalization Strategy
Wireless signals traveling through a **3GPP FlatFadingChannel** experience violent amplitude scaling and random phase rotation on the complex IQ plane. Without mitigation, this completely scrambles data arrays, rendering a standard Convolutional or Dense architecture unable to learn (stuck at a random coin-flip `0.5000` BER). 

This prototype implements **Perfect Channel Equalization**. By dividing out the complex channel coefficients (h) prior to neural ingestion, we eliminate the random phase spin. This stabilizes the **16-QAM constellation grid energy to a strict 1.0 average mapping**, allowing the underlying dense neural layers to discover stable, deterministic coordinate decoding boundaries.

---

## 📁 Repository Structure

*   **[`THEORY_AND_MAPPING_GUIDE.md`](./THEORY_AND_MAPPING_GUIDE.md)** - **Master Onboarding Blueprint**. Defines the 3-Computer Paradigm, wireless engineering terminologies, and handles deep architectural tracking.
*   **`colab/`** - Cloud-based data engineering simulation logic and verified training loop parameters.
*   **`data/`** - *Transient Hardware Cache*. Temporarily houses your downloaded 3GPP channel tensors (`.npy`) for local matrix streaming. Blocked via `.gitignore` to prevent repository binary bloat.
*   **`models/`** - Exported native PyTorch dense model brain state binaries (`.pth`). Manual cloud download asset safely excluded from public version tracking.
*   **`src/`** - Local modular Apple Silicon execution modules accelerated via `Metal (MPS)`.


```text
.
├── colab/                         # Cloud CUDA generation & training scripts
├── data/                          # Exported 3GPP channel radio wave & bit labels (.npy)
├── models/                        # Exported native PyTorch model brain states (.pth)
└── src/                           # Local modular Apple Silicon execution modules
    ├── model.py                   # 6G Dense Linear Neural Decoder definition
    ├── mac_inference.py           # Eager baseline deployment execution harness
    └── mac_inference_optimized.py # Graph-Compiled AOT optimization module
```

---

## 📊 Performance, Graph Optimization & Hardware Profiling Summary

The neural receiver parameters were optimized using an **NVIDIA T4 CUDA Tensor Core GPU** inside a cloud simulation instance. Local edge evaluation, latency profiling, and architectural benchmarking were carried out via **PyTorch MPS (Metal Performance Shaders)** utilizing Apple Silicon's unified memory bandwidth.

### Local Edge Latency Evolution (M4 Core Execution)

To bridge the gap between initial prototypes and strict real-time 6G slot processing deadlines, this repository profiles the neural receiver across three distinct optimization phases:

1. **Initial Un-optimized Build:** High-overhead loop execution constraints.
2. **Eager Modular Baseline (`mac_inference.py`):** Structured matrix pipelines dispatched sequentially.
3. **Graph-Compiled Optimization (`mac_inference_optimized.py`):** Ahead-of-Time (AOT) graph fusion utilizing **TorchDynamo** to eliminate host-to-device tracking overhead and cache intermediate layer data directly on-chip.


| Profiling Metric | Initial Build | Eager Baseline | Graph-Compiled Loop |
| :--- | :--- | :--- | :--- |
| **Target Execution Device** | Apple M4 (`mps`) | Apple M4 (`mps`) | Apple M4 (`mps`) |
| **Total Radio Payload Processed** | **229,376 bits** | **229,376 bits** | **229,376 bits** |
| **End-to-End Latency** | `492.818 ms` | `55.490 ms` | **6.686 ms** |
| **Optimization Gains** | Baseline | ⚡ ~9x via Matrix Optimization | **🚀 ~73x Cumulative Speedup** (8.3x via Graph Fusion) |
| **Verified Bit Error Rate (BER)** | `0.0000` | `0.0000` | **0.0000 (100% Signal Recovery)** |
| **Final Cloud Training Loss** | `0.0156` | `0.0156` | `0.0156` |

---
## 🛠️ Step-by-Step Local Deployment Workflow

### 1. Initialize and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy
```

### 2. Execute Local Inference
```bash
cd src
python3 mac_inference.py
```


## 🎯 The Core Engineering Problems Solved

This repository provides a concrete solution to the high barrier to entry and resource bottlenecks associated with prototyping AI-native physical layers (L1) in next-generation networks. By establishing a **Hybrid Hardware Workflow**, it decouples heavy electromagnetic channel modeling from localized edge execution.

### 🛑 1. Shattering the "NVIDIA/Linux-Only" Bottleneck
* **The Constraint:** Advanced 6G link-level simulators—specifically **NVIDIA Sionna 2.0**—strictly require a dedicated Linux/Windows host backed by CUDA-enabled NVIDIA graphics cards to compute dense 3GPP wave equations. 
* **The Solution:** This project utilizes a **Hybrid Data Bridge**. By running the heavy physical channel math dynamically inside a cloud instance, the pipeline serializes and downloads synchronized tensor matrices (`.npy`) and weight matrices (`.pth`) locally. This allows developers to prototype 6G AI-RAN architectures on consumer-grade computers without complex system-level driver constraints.

### 🛑 2. Eliminating Edge Model Bloat via Smart Partitioning
* **The Constraint:** Standard end-to-end neural receivers often rely on heavy, parameters-bloated 2D Convolutional Layers (CNNs) that struggle to meet strict real-time slot processing deadlines.
* **The Solution:** This architecture implements a **partitioned hybrid design**. Instead of forcing an AI model to waste processing cycles calculating physical channel rotations natively, classical matrix calculus handles **Perfect Equalization ($y/h$)** during the cloud data-engineering phase. Because the phase spin is pre-rectified, the local edge model is stripped down to an ultra-lightweight, 3-layer `nn.Linear` Coordinate Decoder that maps clean 16-QAM coordinates to bits with a flawless **0.0000 Bit Error Rate (BER)**.

### 🛑 3. Defeating Python Runtime Overhead
* **The Constraint:** Running PyTorch layers sequentially in traditional Eager Mode introduces massive interpreter tracking overhead, causing local inference times to lag at **55.490 ms**.
* **The Solution:** The deployment engine leverages **Ahead-of-Time (AOT) Graph Compilation** powered by `TorchDynamo`. It fuses multi-layer mathematical graphs directly onto the silicon, dropping processing execution time down to an ultra-fast **6.686 ms** using local **Apple Metal Performance Shaders (MPS)**.

---


## 🔍 What is This Problem and Workflow Valid For?

This project does not attempt to solve an unmapped wireless channel estimation mystery. Instead, it provides a highly valid, production-ready solution to a critical **software engineering, resource accessibility, and infrastructure bottleneck** in next-generation network prototyping. 

The architecture is mathematically and structurally valid for three major industry use cases:

### 1. Demarginalizing 6G Research (Shattering the Hardware Wall)
Advanced 6G physical layer simulation toolkits—most notably **NVIDIA Sionna**—are strictly bound to Windows/Linux environments running native NVIDIA CUDA hardware configurations. 
* **The Problem:** Software engineers, mobile application developers, and researchers operating on local **Apple Silicon macOS** machines are completely locked out of executing live link-level 3GPP physics loops at their desks.
* **The Validity:** This project establishes a valid, replicable asynchronous data loop. By offloading heavy physical simulations to transient cloud GPU instances, stripping the phase variance via pre-processing, and exporting synchronized `.npy` and `.pth` binaries, it creates a hardware-agnostic playground. It proves you can develop and validate 6G physical layer components without local enterprise server racks.

### 2. Validating Real-Time L1 Slot Deadlines (Sub-10ms Constraints)
In live Radio Access Networks (RAN), Physical Layer (Layer 1) processing elements must obey strict hardware timing boundaries. Signal blocks must be processed within **cellular slot deadlines—typically under 10 milliseconds**—or the base station experiences downlink buffer bloat and drops the connection.
* **The Problem:** Standard Python deep learning execution (Eager Mode) introduces immense interpreter overhead, lagging at an unusable **~55.490 ms** on edge hardware.
* **The Validity:** Your local profiling runs proved that applying **Ahead-of-Time (AOT) Graph Compilation (`torch.compile`)** fused the underlying neural layers into a singular compiled machine-code graph. By running directly inside the Mac's unified memory cache via Metal Performance Shaders (MPS), execution plummeted to **6.686 ms**. This mathematically validates that lightweight edge architectures can comfortably beat live cellular timing constraints on consumer desktop silicon.

### 3. Creating an Isolated Sandbox for Edge Optimization (TinyML & Quantization)
Before telecommunication firms flash an AI model onto a physical base station chip or a smartphone's baseband modem, they must profile how the underlying silicon handles high-dimensional tensor operations under strict control constraints.
* **The Problem:** If the data stream contains unpredictable, continuously shifting multi-path fading spins, engineers cannot accurately isolate whether a spike in Bit Error Rate (BER) is caused by a hardware quantization error or environmental noise.
* **The Validity:** By utilizing a perfect mathematical equalization block as a pre-filtering step, you create a pristine geometric control environment. The model functions as a pure coordinate-to-bit classifier. This isolated setup is perfectly valid for stress-testing higher-order layouts (like 64-QAM/256-QAM) or executing **low-precision quantization steps (FP16/INT8)** to measure exactly how memory bandwidth changes without environmental noise muddying the hardware profile.

---

## 🛠 Target Use Cases & Applications

This repository functions as an extensible blueprint for several real-world telecommunication R&D workflows:
* **Offline Edge Latency Profiling:** Benchmark how fast a specific neural layer topology processes high-dimensional wireless tensors without drawing power from expensive cloud clusters or tying up software-defined radio (SDR) hardware arrays.
* **Apple Silicon AI-RAN Benchmarking:** A rare example demonstrating how to properly format, permute, and compile 5G NR/6G high-dimensional tensor shapes directly into Apple's unified memory architecture for rapid local prototyping.
* **Constellation Boundary Geometry Mapping:** Because the core network functions as a geometric coordinate classifier, researchers can use this framework to instantly visualize how varying thermal background noise scales ($N_0$) warp the decision boundaries of high-order modulation schemes.


### 🚀 Scaling Blueprint: Upgrading to 64-QAM / 256-QAM

The current architecture is configured for a **16-QAM** constellation envelope ($2^4$ unique bits per symbol). To scale this framework to support advanced high-capacity 6G profiles, implement the following cascading structural adjustments across the files:

1. **Modify Simulation Constants (`colab/` & `src/model.py`):**
   * Change `num_bits_per_symbol = 6` (for 64-QAM) or `8` (for 256-QAM).
2. **Recompute the Total Array Footprint:**
   * Scaling the modulation increases the total row bits size dynamically. Update the batch calculator constraint to match: 
   * `total_bits_per_item = num_subcarriers * num_symbols * num_bits_per_symbol`
3. **Automatic Network Head Adaptation:**
   * Thanks to the modular code layout, the dense decoding stack output dimension automatically locks to your configured `num_bits_per_symbol` target. The network head will autonomously expand its boundary classifiers to match the denser constellation coordinate clustering.

💡 **Why We Do This (The Telecom Context):** Higher-order modulations like 64-QAM and 256-QAM pack more data bits into every single radio wave symbol, exponentially increasing the total network throughput. For a neural network, this transforms a simple 16-point geometric puzzle into a dense, multi-cluster classification problem. Testing this scaling shows whether a lightweight model can still successfully draw distinct geometric decision boundaries as the data clusters get crowded closer together under high thermal noise.

---

## 💻 System Technical Prerequisites

To execute this hybrid workflow seamlessly, confirm your physical and cloud runtime configurations align with the following boundaries:

### ☁ Cloud Ingestion Layer
* **Simulation Framework:** NVIDIA Sionna 2.0.1+ (Requires CUDA runtime drivers)
* **Compute Hardware Target:** NVIDIA T4, L4, A100, or H100 Tensor Core Instance
* **Python Runtime Environment:** Python 3.10 to Python 3.12 (Sionna compiled constraint)

💡 **Why This is Required:** NVIDIA Sionna is built natively on top of TensorFlow and customized CUDA C++ kernels to simulate real-world radio propagation (like multi-path ray tracing and fading) at physical light speed. Because it relies heavily on NVIDIA Tensor Cores to run these 3GPP physics equations, this layer *must* be offloaded to a cloud instance with CUDA-capable hardware.

### 💻 Local Deployment Layer
* **Inference Library:** Native PyTorch 2.2+ (With MPS backend bindings mapped natively)
* **Compute Hardware Target:** Apple Silicon (M1/M2/M3/M4 Series Architecture) with Unified Memory
* **Optimization Prerequisite:** `torch.compile()` execution requires `setuptools` and a working local C++ toolchain installation (Xcode Command Line Tools) to process Ahead-of-Time Graph Fusion via TorchDynamo.

💡 **Why This is Required:** To drop processing latency down to **6.686ms** without an NVIDIA card, we utilize Apple's **Metal Performance Shaders (MPS)** backend. By using `torch.compile()`, a local C++ compiler strips away Python's slow, line-by-line interpreter loop and creates a fused, compiled machine-code graph. This ensures the model runs directly inside the Mac's ultra-fast Unified Memory cache to meet real-time 6G cellular slot deadlines.


---

## 🔮 Future Horizon: What This Serves in the Coming Era

This framework is not just a static blueprint for offline profiling; it is a foundational sandbox for the next wave of software-defined telecommunication breakthroughs. Maintaining this hybrid architecture serves as an entry point for several emerging paradigms:

### 📡 1. The Gateway to Real-Time SDR Deployments
Because the local edge deployment layer compiles down to a sub-10ms processing loop using PyTorch and Apple Silicon, this codebase is structurally primed for **Software-Defined Radio (SDR) integration**. By swapping the static `.npy` trace files for a live C++ wrapper API (like GNU Radio, UHD drivers, or OpenAirInterface), this neural receiver can be plugged directly into physical USRP or BladeRF hardware, morphing it into a live, over-the-air 6G AI-RAN prototype base station.

### 🌐 2. Native Foundation for Digital Twin Optimization
As 6G expands into localized, site-specific configurations, base stations will use real-time 3D Ray Tracing (**Sionna RT**) to build live "Digital Twins" of physical environments (e.g., specific factories, ports, or downtown city grids). This repo provides the exact target model architecture required for that future workflow. Engineers can continuously update the cloud simulation loop with site-specific 3D spatial geometry data, generate highly accurate localized weights, and flash them straight down to the edge hardware to keep pace with changing structural layouts.

### ⚡ 3. Hardware-Agnostic Edge Benchmarking (ONNX & TinyML)
The deliberate decoupling of the data pre-processing logic from the model architecture turns this repo into a pristine benchmarking harness for embedded computing. The lightweight, linear-dense layer setup is a prime candidate for **ONNX compilation and CoreML translation**. This opens the door to testing and profiling 6G receiver footprints across an expansive matrix of ultra-low-power edge hardware, from IoT nodes to consumer mobile devices running custom neural processors.

