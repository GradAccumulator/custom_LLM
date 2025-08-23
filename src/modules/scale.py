import torch
from .base_module import BaseModule
from src.utils.types import ListLike

class Scale(BaseModule):
    def __init__(self,value:ListLike|float,*, shape=None, learnable=False, dtype=torch.float32, device=None):
        super().__init__(device, dtype)
        self.learnable = learnable
        self.shape = shape
        self.value = value
        if isinstance(value, ListLike):
            self.scale = torch.tensor(value, device=self.device, dtype=self.dtype, requires_grad=self.learnable)
        elif self.shape is not None:
            self.scale = torch.full(self.shape, value, device=self.device, dtype=self.dtype, requires_grad=self.learnable)
        else:
            self.scale = torch.tensor(value, device=self.device, dtype=self.dtype, requires_grad=self.learnable)
        if self.learnable:
            self.parameters = self._parameters
            self.clone = self._clone
            self.init_like = self._init_like
    
    def forward(self, x:torch.Tensor):
        return x*self.scale.to(x.dtype)
    
    def __call__(self, x):
        return self.forward(x)
    def __mul__(self, other):
        return self.forward(other)
    def __rmul__(self, other):
        return self.forward(other)
    
    def _parameters(self):
        return [self.scale]
    
    def to(self, device=None, dtype=None):
        if (device is None and dtype is None) or (self.device == device and self.dtype == dtype):
            return self
        scale = self._init_like()
        scale.device = device if device is not None else self.device
        scale.dtype = dtype if dtype is not None else self.dtype
        scale.scale = self.scale.detach().to(device=scale.device, dtype=scale.dtype)
        if self.learnable:
            scale.scale = scale.scale.requires_grad_()
        return scale
    
    def _clone(self):
        scale = Scale(value=self.value, shape=self.shape, learnable=self.learnable, dtype=self.dtype, device=self.device)
        scale.scale = self.scale.detach().clone()
        if self.learnable:
            scale.scale = scale.scale.requires_grad_()
        return scale
    
    def _init_like(self):
        return Scale(value=self.value, shape=self.shape, learnable=self.learnable, dtype=self.dtype, device=self.device)
    
    def state_dict(self):
        return {
            'scale':self.scale.detach().cpu(),
            'scale_grad':self.scale.grad.detach().cpu() if self.scale.grad is not None else None
        }
    def load_state(self, state):
        self.state_key_check_with_raise(state.keys(), ['scale', 'scale_grad'], Scale, 'load_state')
        with torch.no_grad():
            if self.learnable:
                self.scale.copy_(state['scale'].to(self.device, self.dtype).requires_grad_())
                if state['scale_grad'] is not None:
                    self.scale.grad.copy_(state['scale_grad'].to(self.device, self.dtype))
            else:
                self.scale.copy_(state['scale'].to(self.device, self.dtype))
                