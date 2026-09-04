# Sionna Neural Receiver: Hybrid 6G AI-RAN Prototype

[![Python 3.12+](https://shields.io)](https://python.org)
[![PyTorch Native](https://shields.io)](https://pytorch.org)
[![NVIDIA 6G Ecosystem](https://shields.io)](https://nvidia.com)

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

*   **`colab/`** - Cloud-based simulation logic. Contains [Interactive Google Colab Notebook Workflow Here](https://colab.research.google.com/drive/1jN3KRhwKP84HJQnQjMxtH6UOdlyfnIVb#scrollTo=yn1SJDCLBX-K) and corresponding pipeline execution briefs.
*   **`data/`** - Exported 3GPP channel radio wave tensors & ground-truth bit labels (`.npy`).
*   **`models/`** - Exported native PyTorch dense model brain state binaries (`.pth`).
*   **`src/`** - Local modular Apple Silicon execution modules accelerated via `Metal (MPS)`.

```text
.
├── colab/               # Cloud CUDA generation & training scripts
├── data/                # Exported 3GPP channel radio wave & bit labels (.npy)
├── models/              # Exported native PyTorch model brain states (.pth)
└── src/                 # Local modular Apple Silicon execution modules
    ├── model.py         # 6G Dense Linear Neural Decoder definition
    └── mac_inference.py # Core local hardware profiling & benchmark harness
```

---

## 📊 Performance & Hardware Profiling Summary

The model weights were trained using an **NVIDIA T4 CUDA Tensor Core GPU** inside a cloud simulation instance. Local evaluation and tracking were carried out via **PyTorch MPS (Metal Performance Shaders)** utilizing Apple Silicon's unified memory bandwidth.

### Local Inference Metrics (M4 Core Execution)

| Metric | Benchmark Result |
| :--- | :--- |
| **Target Core Processing Device** | Apple M4 Neural/GPU (via `torch.device("mps")`) |
| **Total Radio Payload Processed** | **229,376 bits** (64 subcarriers × 14 symbols × 4 bits/symbol) |
| **End-to-End Hardware Latency** | **492.818 ms** |
| **Verified Bit Error Rate (BER)** | **0.0000 (Perfect 100% Signal Recovery)** |
| **Final Cloud Training Loss** | **0.0156** |

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
