import torch
from src.modules.embedding import (Embedding, Module)

class PositionalEmbedding(Module):
    def __init__(self, d_in, d_out,*,device='cpu', table_dtype=torch.float32, use_cached_arange=True, cached_arange_dtype=torch.long):
        super().__init__(device, table_dtype, d_in=d_in, d_out=d_out, device=device, table_dtype=table_dtype, use_cached_arange=use_cached_arange, cached_arange_dtype=cached_arange_dtype)
        
        self.d_in = d_in
        self.d_out = d_out
        self.use_cached_arange = use_cached_arange
        self.cached_arange_dtype = cached_arange_dtype
        
        self.cached_arange = torch.arange(self.d_in, device=self.device, dtype=self.cached_arange_dtype)
        self.embedding = Embedding(self.d_in, self.d_out, device=self.device, dtype=self.table_dtype)
    
    def forward(self, x:torch.Tensor):
        if self.use_cached_arange and x.size(-2) == self.d_in:
            return x + self.embedding(self.cached_arange.to(torch.long)).to(x.dtype)
        else:
            return x + self.embedding(torch.arange(0, x.size(-2), device=self.device)).to(x.dtype)
    
    @property
    def table_dtype(self):
        return self.dtype