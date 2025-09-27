import torch
from src.nlp.embedding import Embedding
from src.utils.torch_utils import 보이면_바로_교체하기

class PositionalEmbedding(Embedding):
    def __init__(self, d_in, d_out,*,device='cpu', dtype=torch.float32):
        super().__init__(d_in, d_out, device=device, dtype=dtype)
        self.cached_arange = torch.arange(self.d_in, device=self.device, dtype=torch.long)
    
    def forward(self, x:torch.Tensor):
        if x.size(-2) != self.d_in:
            return (x+super().forward(torch.arange(x.size(-2), device=self.device)))
        
        return (x+super().forward(self.cached_arange).to(x.device, x.dtype))