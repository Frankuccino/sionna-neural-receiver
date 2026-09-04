import torch
import torch.nn as nn


class NeuralReceiverM4(nn.Module):
    def __init__(self, num_bits_per_symbol=4):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_bits_per_symbol),
        )

    def forward(self, y):
        subcarriers = y.shape[-2]
        symbols = y.shape[-1]

        y_flattened = y.view(64, -1, subcarriers, symbols)
        y_clean = torch.mean(y_flattened, dim=1)

        y_real = torch.real(y_clean)
        y_imag = torch.imag(y_clean)

        x = torch.stack([y_real, y_imag], dim=-1)
        llr = self.decoder(x)

        return llr.reshape(64, -1)
