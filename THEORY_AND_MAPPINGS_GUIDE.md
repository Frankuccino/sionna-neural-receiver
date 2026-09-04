# 🗺️ 6G AI-RAN Master Theory, Domain Separations, & Common Misconceptions

Welcome to the architectural roadmap for the **Sionna Neural Receiver Sandbox**. This comprehensive onboarding guide is engineered for developers, telecom researchers, and deep learning practitioners who want to cross-reference and master the exact code layout, telecommunication physics, and graph compilation steps implemented in this repository.

---

## 🏗️ 1. Domain Separation: The Three-Computer Paradigm

A major friction point for software developers entering the wireless space is mixing up **where the code physically executes** versus **what physics it is simulating**. This project strictly separates these environments into three isolated tiers:

```
┌────────────────────────────────┐       ┌────────────────────────────────┐
│   1. THE SIMULATION DOMAIN     │       │     2. THE AI BRAIN DOMAIN     │
│  - NVIDIA Sionna 2.0 Engine    │       │  - Native PyTorch Linear Stack │
│  - Cloud CUDA VRAM Environment │  ───▶ │  - Cloud-Trained Weights (.pth)│
│  - 3GPP Physics Emulation      │       │  - Bounded Coordinate Mapping  │
└────────────────────────────────┘       └────────────────────────────────┘
                │
                ▼
┌────────────────────────────────┐
│     3. THE COMPILER DOMAIN     │
│  - Apple Silicon GPU Backend   │
│  - Ahead-of-Time Graph Fusion  │
│  - Sub-Millisecond L1 Target   │
└────────────────────────────────┘
```

1. **The Simulation Domain (NVIDIA Sionna Core):** This handles the complex physics of electromagnetism. It generates random bit streams, wraps them into radio wave shapes, and warps them using fading math. This must run on an **NVIDIA Cloud GPU** because its underlying matrix engines are built exclusively on Linux CUDA layers.
2. **The AI Brain Domain (PyTorch Layer):** This is the neural model architecture. It handles the raw classification tasks, converting real and imaginary coordinate snapshots into standard Log-Likelihood Ratio matrices. It saves these calculations as compact parameter weights (`.pth`).
3. **The Deployment Compiler Domain (Apple Silicon Core):** This handles local, real-time edge processing. It strips away all the heavy cloud simulation machinery, downloads the raw data traces (`.npy`), loads the trained weights (`.pth`), and uses `torch.compile(backend="mps")` to optimize the math for your Mac's shared memory layout.

---

## 🛠️ 2. Core Telecommunication Glossary

To navigate this codebase fluently, you must master the fundamental wireless concepts being modeled inside the data pipelines:

- **3GPP (3rd Generation Partnership Project):** The global governing body that establishes standard cellular rules and mathematical equations. Sionna is programmed to match these official specifications exactly.
- **16-QAM (Quadrature Amplitude Modulation):** A method for mapping digital data onto a radio wave. A 16-QAM setup maps a flat stream of bits into groups of 4 bits per symbol (2⁴ = 16). This assigns each group a distinct coordinate position on a 2D grid called the **IQ Plane** (composed of an In-phase axis and a Quadrature phase axis).
- **Flat Fading Channels:** A physical layer distortion that occurs when a radio wave bounces off buildings or trees. This alters the signal's size (amplitude) and violently spins its orientation (phase) on the IQ plane, causing severe errors if unmitigated.
- **CSI (Channel State Information) & Equalization:** The mathematical blueprint of how a radio channel distorted the signal. In this project, we implement **Perfect Zero-Phase Equalization** by dividing the received signal (Y) by the channel footprint (H). This stops the constellation from spinning, giving the AI clear coordinates to decode.
- **LLR (Log-Likelihood Ratio):** The network's soft-decision output. Instead of guessing a hard 1 or 0 instantly, the model outputs a floating-point probability score. A large positive number indicates a confident 1, a large negative number indicates a confident 0, and a score near 0 indicates a highly uncertain guess.

---

## 🚨 3. Major Misconceptions & The Correct Mental Path

### ❌ Misconception A: "Wireless signals look like smooth, spatial visual images, so we must use a CNN."
- **The Wrong Path:** Assuming that because an OFDM wireless grid has axes for frequency and time, a 2D Convolutional Network (`nn.Conv2d`) can treat it like a digital image to filter out background noise.
- **The Right Path:** Digital bits modulated onto a 16-QAM constellation are mapped to discrete, isolated coordinates. A sliding CNN kernel can blur these distinct boundaries together, which breaks the precision needed to recover the underlying data bits. This project proves that an element-wise **Linear Dense Decoder Layer Stack** is the correct architecture, mapping discrete coordinates to raw bits with a perfect 0.0000 Bit Error Rate (BER).

### ❌ Misconception B: "The model's random guessing (0.5000 BER) is a bug in the code syntax."
- **The Wrong Path:** Searching for syntax errors or constantly rewriting your layer code because your training logs show the model is stuck flipping a coin.
- **The Right Path:** The code syntax is perfectly valid. The problem is a **data engineering flaw**. If you pass raw, unequalized radio waves into an AI without providing the underlying channel blueprint (H), the constellation spins unpredictably every step. This makes the data look like flat random noise. Introducing an equalization step solves this constraint instantly.

### ❌ Misconception C: "We need an expensive desktop-class NVIDIA GPU to build, deploy, and benchmark 6G models."
- **The Wrong Path:** Believing you are blocked from 6G AI development because your local computer runs on Apple Silicon or lacks local Linux CUDA capabilities.
- **The Right Path:** You can implement a **Hybrid Edge Architecture**. Use a free cloud GPU instance to run heavy physical simulations and train model configurations via Sionna. Then, download the compact weight outputs (`.pth`) and data streams (`.npy`) locally. This allows you to build a lean, lightning-fast edge inference harness on an M4 Mac Mini using Apple's local performance shaders.

---

## 🗺️ 4. Master Roadmap to Absolute Mastery

To fully master the code layout and concepts in this repository, execute your learning path in this specific sequence:

```
[ Step 1: colab/README.md ] ──▶ [ Step 2: models/README.md ] ──▶ [ Step 3: src/README.md ]
```

1. **Read `/colab/README.md`**: Master how cloud data engineering, 3GPP fading channels, and constellation mapping physics function on NVIDIA CUDA systems.
2. **Read `/models/README.md`**: Analyze how neural networks map 2D spatial inputs down to discrete bit outputs, and explore cross-platform weight serialization formats.
3. **Read `/src/README.md`**: Evaluate eager execution mode constraints against graph compilation paths, and understand why dropping local latency to **6.686 ms** is critical for 6G slot boundaries.

> 📝 **Architectural Note on `/data/`:** The `/data` directory acts strictly as a raw, transient cache for streaming the local NumPy tensors (`.npy`). Because it contains static binary storage arrays rather than source logic, it is purposefully excluded from the theory mapping track and blocked via `.gitignore` to prevent tracking binary file bloat.
