import torch
from .activation_function import ActivationFunction
from src.utils.torch_utils import device_dtype_check

class ReLU(ActivationFunction):
    def __init__(self):
        pass
    def forward(self, x:torch.Tensor):
        return torch.where(x>0, x, torch.tensor(0))
    
    def __repr__(self):
        return "ReLU()"
    
class LeakyReLU(ActivationFunction):
    def __init__(self, a=0.1):
        self.a = a

    def forward(self, x:torch.Tensor):
        return torch.where(x>0, x, x*self.a)
    
    def __repr__(self):
        return f"LeakyReLU(a={self.a})"

from src.modules.base_module import BaseModule
class PReLU(BaseModule, ActivationFunction):
    def __init__(self, a_init=0.1, device='cpu', dtype=torch.float32):
        self.device = device
        self._dtype = dtype
        self._a = torch.tensor(a_init, requires_grad=True, dtype=dtype, device=device)
        self.a_init = a_init

    def forward(self, x:torch.Tensor):
        return torch.where(x>0, x, x*self._a)
    
    @property
    def a(self):
        return self._a
    
    def __repr__(self):
        return f'PReLU(a={self.a.tolist()})'