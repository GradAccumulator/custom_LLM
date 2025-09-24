import torch
from src.modules import (
    Sequential,
    Linear,
    ReLU,
    Sigmoid
)

class VAE:
    def __init__(self, d_in, hidden_dim, z_dim, device='cpu', dtype=torch.float32):
        self.d_in = d_in
        self.d_latent = hidden_dim
        self.z_dim = z_dim
        
        self.encoder = Sequential(
            Linear(d_in, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, z_dim*2)
        )
        
        self.decoder = Sequential(
            Linear(z_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, d_in),
        )
        
        self.encoder = self.encoder.to(device=device, dtype=dtype)
        self.decoder = self.decoder.to(device=device, dtype=dtype)
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor):
        return mu + logvar.mul(0.5).exp()*torch.randn_like(logvar)
    
    def forward(self, x:torch.Tensor):
        z = self.encoder(x)
        mu,logvar = z.split(2, -1)
        sample = self.reparameterize(mu, logvar)
        out = self.decoder(sample)
        return out
    
    def parameters(self):
        return self.encoder.parameters()+self.decoder.parameters()
    def parameters_as_dict(self):
        params = {}