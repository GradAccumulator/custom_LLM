import torch
from src.modules.base_module import Module

class Embedding(Module):
    def __init__(self, d_in, d_out, device='cpu', dtype=torch.float32):
        super().__init__(device, dtype, d_in=d_in, d_out=d_out, device=device, dtype=dtype)
        self._d_in = d_in
        self._d_out = d_out
        self.table = torch.randn((d_in, d_out), dtype=dtype, device=device, requires_grad=True)
        self.init_parameter(self.table, 'xavier')

    def forward(self, x:torch.Tensor):
        return self.table[x]
    
    @property
    def d_in(self):
        return self._d_in
    
    @property
    def d_out(self):
        return self._d_out