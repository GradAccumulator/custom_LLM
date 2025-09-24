import torch
from src.modules import (
    Linear,
    Sequential,
    ReLU
)

class AE:
    def __init__(self, input_dim, hidden_dim, output_dim, device='cpu', dtype=torch.float32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim)
        )

        self.decoder = Sequential(
            Linear(output_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, input_dim)
        )

        self.encoder = self.encoder.to(device=device, dtype=dtype)
        self.decoder = self.decoder.to(device=device, dtype=dtype)