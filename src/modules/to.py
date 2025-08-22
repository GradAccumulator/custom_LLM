import torch


class To:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
    def __call__(self, x):
        return x.to(self.device, self.dtype)
    def to(self, device=None, dtype=None):
        return To(device=device or self.device, dtype=dtype or self.dtype)