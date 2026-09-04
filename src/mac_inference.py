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

# Load the trained model weights straight into Mac memory
model = NeuralReceiverM4(num_bits_per_symbol=4)
model.load_state_dict(
    torch.load("../models/neural_receiver_6g_weights.pth", map_location=device)
)
model.to(device)
model.eval()

# Profile Local Latency Performance
start_time = time.perf_counter()

with torch.no_grad():
    predicted_llrs = model(y_equalized)
    hard_decisions = (predicted_llrs > 0).float()
    bit_errors = torch.not_equal(hard_decisions, tx_bits).float().mean().item()

end_time = time.perf_counter()
execution_ms = (end_time - start_time) * 1000

print("\n--- M4 Mac Mini Inference Summary ---")
print(f"Total Bits Processed  : {tx_bits.numel():,} bits")
print(f"Hardware Latency      : {execution_ms:.3f} milliseconds")
print(f"Verified Local BER    : {bit_errors:.4f}")
