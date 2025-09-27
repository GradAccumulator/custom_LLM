import torch
from .base_module import BaseModule

class LayerNorm(BaseModule):
    def __init__(self, dim,*,eps=5e-5, device='cpu', dtype=torch.float32):
        super().__init__(device, dtype, dim=dim, eps=eps, device=device, dtype=dtype)
        self.eps = eps
        self.dim = dim
        self.gamma = torch.ones(self.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        self.beta = torch.zeros(self.dim, device=self.device, dtype=self.dtype, requires_grad=True)

    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        out = (x-mean)/(self.eps + std)
        out = self.gamma.to(x.dtype)*out + self.beta.to(x.dtype)
        return out
        
    def __repr__(self):
        return f'LayerNorm(dim={self.dim}, eps={self.eps})'