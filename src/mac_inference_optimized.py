import torch
import numpy as np
import time
from model import NeuralReceiverM4  # Clean modular import

# Initialize Apple Silicon Core Context
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Target Hardware Acceleration: {device}")

# Load Arrays and Weights from their structured paths
y_equalized_np = np.load("../data/y_equalized_test.npy")
tx_bits_np = np.load("../data/tx_bits_test.npy")

y_equalized = torch.from_numpy(y_equalized_np).to(torch.complex64).to(device)
tx_bits = torch.from_numpy(tx_bits_np).float().to(device)

# Instantiate the model and load weights
model = NeuralReceiverM4(num_bits_per_symbol=4)
model.load_state_dict(
    torch.load("../models/neural_receiver_6g_weights.pth", map_location=device)
)
model.to(device)
model.eval()

# --- FIX: INJECT TORCH COMPILATION GRAPH OPTIMIZATION ---
print("\nCompiling model graph via torch.compile(backend='mps')...")
compiled_model = torch.compile(model)

# Warmup run (Forces the MPS compiler to build the optimized kernel graph)
print("Executing compilation warmup slot...")
with torch.no_grad():
    _ = compiled_model(y_equalized)

# Ensure Apple Silicon finishes compiling before starting the clock
if device.type == "mps":
    torch.mps.synchronize()

print("Compilation complete! Running optimized benchmark...")

# Profile Local Latency Performance
start_time = time.perf_counter()

with torch.no_grad():
    predicted_llrs = compiled_model(y_equalized)

    # Ensure all asynchronous GPU operations finish before stopping the clock
    if device.type == "mps":
        torch.mps.synchronize()

    hard_decisions = (predicted_llrs > 0).float()
    bit_errors = torch.not_equal(hard_decisions, tx_bits).float().mean().item()

end_time = time.perf_counter()
execution_ms = (end_time - start_time) * 1000

print("\n--- M4 Mac Mini OPTIMIZED Inference Summary ---")
print(f"Total Bits Processed  : {tx_bits.numel():,} bits")
# This will capture the raw compiled kernel execution speed!
print(f"Hardware Latency      : {execution_ms:.3f} milliseconds")
print(f"Verified Local BER    : {bit_errors:.4f}")
