import torch
from .base_module import BaseModule
from src.utils.types import ListLike
from src.utils.torch_utils import device_dtype_check

class Scale(BaseModule):
    def __init__(self,value:ListLike|float,*, shape=None, learnable=False, dtype=torch.float32, device=None):
        super().__init__(device, dtype, value=value, shape=shape, learnable=learnable, dtype=dtype, device=device)
        self.learnable = learnable
        self.shape = shape
        self.value = value
        if isinstance(value, ListLike):
            self.scale = torch.tensor(value, device=self._device, dtype=self._dtype, requires_grad=self.learnable)
        elif self.shape is not None:
            self.scale = torch.full(self.shape, value, device=self._device, dtype=self._dtype, requires_grad=self.learnable)
        else:
            self.scale = torch.tensor(value, device=self._device, dtype=self._dtype, requires_grad=self.learnable)
    
    def forward(self, x:torch.Tensor):
        return x*self.scale.to(x.dtype)
    def __mul__(self, other):
        return self.forward(other)
    def __rmul__(self, other):
        return self.forward(other) 
    
    def __repr__(self):
        return (
            f'Scale(\n'
            f'    value={self.value},\n'
            f'    shape={self.shape},\n'
            f'    learnable={self.learnable},\n'
            f'    dtype={self._dtype},\n'
            f'    device={self._device}\n'
            f')'
        )