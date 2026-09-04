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
