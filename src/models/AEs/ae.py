import torch
from src.modules import (
    Linear,
    Sequential,
    ReLU
)
from src.modules.base_module import Module

class AE(Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cpu', dtype=torch.float32):
        super().__init__(device, dtype, input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device, dtype=dtype)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = latent_dim

        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, latent_dim)
        )

        self.decoder = Sequential(
            Linear(latent_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, input_dim)
        )

        self.encoder.to(device=device, dtype=dtype)
        self.decoder.to(device=device, dtype=dtype)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return decoder_out