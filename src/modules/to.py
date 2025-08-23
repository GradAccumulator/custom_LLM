import torch


class To:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
    def forward(self, x):
        return x.to(self.device, self.dtype)
        
    def __call__(self, x):
        return self.forward(x)
    def to(self, device=None, dtype=None):
        return To(device=device or self.device, dtype=dtype or self.dtype)