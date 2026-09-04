# 🧠 Model Architecture & Weight Serialization Parameters

This directory contains the serialized binary brain states of the trained 6G Neural Receiver. By saving only the learned weights (`.pth`) rather than full software execution environments, the model architecture achieves high cross-platform portability—allowing parameters to be optimized via CUDA on cloud server nodes and seamlessly deployed on Apple Silicon edge devices.

> ⚠️ **Data Sourcing Notice:** The target weight file `neural_receiver_6g_weights.pth` is a heavy binary artifact generated directly inside the cloud environment runtime memory. **You must download it manually from your Google Colab instance sidebar file panel and drop it straight into this local `/models` directory.** To follow open-source engineering best practices, this file is explicitly tracked inside `.gitignore` and **is not uploaded to public GitHub repositories** to keep repository storage footprint lightweight.

---

## 📐 Neural Network Dimensionality & Layers

The receiver is structurally configured as an element-wise **Linear Dense Decoder** rather than a spatial Convolutional Network (CNN). Because digital bits are mapped into discrete coordinates on the real and imaginary planes independently, treating each resource grid element as an isolated coordinate vector prevents spatial blurring and ensures deterministic bit recovery.

The sequential layer stack maps down mathematically as follows:

```text
Input Tensor: [64, 896, 2]  --> Real & Imaginary coordinate components
     │
     ├──> nn.Linear(in_features=2, out_features=32) + nn.ReLU()
     ├──> nn.Linear(in_features=32, out_features=32) + nn.ReLU()
     └──> nn.Linear(in_features=32, out_features=4)   --> Log-Likelihood Ratios (LLRs)
     │
Output Tensor: [64, 3584]   --> Reshaped to align with ground-truth bit payloads
```

### Underlying Tensor Parameters

*   **`in_features = 2`**: Represents the complex IQ plane split into two distinct floating-point channels: Real(Y) and Imag(Y) across 896 resource elements per batch row (64 subcarriers × 14 symbols).
*   **Hidden Layers**: Two sequential hidden blocks with 32 nodes each activated via **Rectified Linear Units (ReLU)** to map non-linear boundary channel distortions safely.
*   **`out_features = 4`**: Maps directly to your bits-per-symbol target (M = 4 for a 16-QAM constellation), outputting raw Log-Likelihood Ratios (LLRs) for each discrete subcarrier and temporal time symbol element.

---

## 💾 Serialization & Cross-Platform Hardware Mapping

The network brain state is saved using PyTorch's native dictionary serialization:

*   **File Format:** `neural_receiver_6g_weights.pth` is a serialized Python dictionary (`OrderedDict`) that maps each layer name string to its underlying floating-point bias and weight tensor matrices (**`state_dict`**).
*   **Zero Over-the-Air Overhead:** By saving only the `state_dict`, the binary payload remains tiny (~5.2 KB) because it contains no execution graph logic or layout wrappers.
*   **Cross-Hardware Mapping (`map_location`):** The weights were compiled on a cloud node using an NVIDIA T4 GPU via standard 32-bit floating-point CUDA primitives (`torch.cuda.FloatTensor`). When running edge inference locally, the script intercepts the byte stream and maps it onto Apple Silicon's execution engine via the structural intercept call:
    ```python
    torch.load('neural_receiver_6g_weights.pth', map_location=device) # device = "mps"
    ```

This enables the identical model brain to translate seamlessly between NVIDIA server clusters and Apple M4 Unified Memory blocks without any mathematical conversion errors.
