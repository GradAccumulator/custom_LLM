import torch
from .base_module import BaseModule

class Linear(BaseModule):
    def __init__(self, d_in, d_out,*, device='cpu', dtype=torch.float32, init_method:str='He', use_bias:bool=True):
        self.init_method = init_method.lower()
        self.d_in = d_in
        self.d_out = d_out
        self._use_bias = use_bias
        super().__init__(device=device, dtype=dtype)
        self.reset_parameters()
        
    def reset_parameters(self):
        self._weight = torch.randn((self.d_in, self.d_out), dtype=self.dtype, device=self.device, requires_grad=True)
        super().init_parameter(self.d_in, self.d_out, self._weight, init_method=self.init_method)
        if self._use_bias:
            self._bias = torch.zeros((self.d_out,), dtype=self.dtype, device=self.device, requires_grad=True)
        
    def forward(self, x:torch.Tensor):
        if self._use_bias:
            return x@self._weight.to(x.dtype) + self._bias.to(x.dtype)
        return x@self._weight.to(x.dtype)
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return [self._weight, self._bias] if self.use_bias else [self._weight]
    
    def to(self, device=None, dtype=None):
        new = self.init_like()
        new.device = device if device is not None else self.device
        new.dtype = dtype if dtype is not None else self.dtype
        new._weight = self._weight.detach().to(device=new.device, dtype=new.dtype).requires_grad_()
        if self.use_bias:
            new._bias = self._bias.detach().to(device=new.device, dtype=new.dtype).requires_grad_()
        return new
    
    @property
    def weight(self):
        return self._weight
    @property
    def bias(self):
        return self._bias
    @property
    def use_bias(self):
        return self._use_bias
    
    def clone(self):
        linear = Linear(self.d_in, self.d_out, device=self.device, dtype=self.dtype, init_method=self.init_method)
        linear._weight = self._weight.detach().clone().requires_grad_()
        if self.use_bias:
            linear._bias = self._bias.detach().clone().requires_grad_()
            
        return linear
    
    def init_like(self):
        return Linear(self.d_in, self.d_out, device=self.device, dtype=self.dtype, init_method=self.init_method)
    
    def state_dict(self):
        return {
            'weight': self._weight.detach().cpu(),
            'bias': self._bias.detach().cpu() if self.use_bias else None,
            'weight_grad': self._weight.grad.detach().cpu(),
            'bias_grad': self._bias.grad.detach().cpu() if self.use_bias else None
        }
    
    def load_state(self, state):
        super().state_key_check_with_raise(state.keys(), ['weight', 'bias', 'weight_grad', 'bias_grad'], Linear, 'load_state')
        with torch.no_grad():
            self.weight.copy_(state['weight'].to(self.device, self.dtype).requires_grad_())
            self.weight.grad = state['weight_grad'].to(self.device, self.dtype)
            if self.use_bias:
                self.bias.copy_(state['bias'].to(self.device, self.dtype).requires_grad_())
                self.bias.grad = state['bias_grad'].to(self.device, self.dtype)