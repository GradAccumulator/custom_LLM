import torch
from src.utils.torch_utils import device_dtype_check

class To:
    def __init__(self, *args, **kwargs):
        self._device, self._dtype = device_dtype_check(*args, **kwargs)
        
    def forward(self, x:torch.Tensor):
        return x.to(self.device, self.dtype)
    def __call__(self, x:torch.Tensor):
        return self.forward(x)
    
    @property
    def device(self):
        return self._device
    
    @property
    def dtype(self):
        return self._dtype
    
    def __repr__(self):
        return f'To(device={self.device}, dtype={self.dtype})'