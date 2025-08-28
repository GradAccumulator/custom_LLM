import torch
from src.utils.torch_utils import device_dtype_check

class To:
    def __init__(self, device=None, dtype=None):
        device,dtype = device_dtype_check(device, dtype)
        self.device = device
        self.dtype = dtype
        
    def forward(self, x:torch.Tensor):
        return x.to(self.device or x.device, self.dtype or x.dtype)
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device=None, dtype=None):
        device, dtype = device_dtype_check(device, dtype)
        return To(device=device or self.device, dtype=dtype or self.dtype)
    
    def __repr__(self):
        return f'To(device={self.device}, dtype={self.dtype})'