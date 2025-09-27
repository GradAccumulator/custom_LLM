import torch
from .base_module import BaseModule
from src.utils.torch_utils import 보이면_바로_교체하기
from typing import Optional, Callable, overload, Any

class Linear(BaseModule):
    d_in_ = int
    d_out_ = int
    parameter = torch.Tensor
    parameter_name = str
    
    @overload
    def __init__(self, d_in, d_out,*, device='cpu', dtype=torch.float32, init_method:Callable[[d_in_, d_out_, parameter_name,parameter], Any], use_bias:bool=True):...
    @overload
    def __init__(self, d_in, d_out,*, device='cpu', dtype=torch.float32, init_method:str="He", use_bias:bool=True):...
    @overload
    def __init__(self, d_in, d_out,*, device='cpu', dtype=torch.float32, init_method:None=None, use_bias:bool=True):...
    
    def __init__(
        self, 
        d_in, 
        d_out,
        *, 
        device='cpu', 
        dtype=torch.float32, 
        init_method:Optional[str|Callable[[d_in_, d_out_, parameter_name,parameter], Any]]='He', 
        use_bias:bool=True
    ):
        self.init_method = init_method
        self.d_in = d_in
        self.d_out = d_out
        self._use_bias = use_bias
        super().__init__(device=device, dtype=dtype)
        self.reset_parameters()
        self.to(device, dtype)
    
    def reset_parameters(self):
        self.weight = torch.randn((self.d_out, self.d_in), dtype=self._dtype, device=self._device, requires_grad=True)
        super().init_weights(self.init_method)
        if self._use_bias:
            self.bias = torch.zeros((self.d_out,), dtype=self._dtype, device=self._device, requires_grad=True)
    
    def forward(self, x:torch.Tensor):
        if self._use_bias:
            return x@self.weight.T.to(x.dtype) + self.bias.to(x.dtype)
        return x@self.weight.T.to(x.dtype)
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device=None, dtype=None):
        device,dtype = 보이면_바로_교체하기(device, dtype)
        if(self._dtype==dtype or dtype is None) and (self._device==device or device is None):
            return self
        new = self.init_like()
        device = device or self._device
        dtype = dtype or self._dtype
        new.weight = self.weight.detach().to(device=device, dtype=dtype).requires_grad_()
        if self.use_bias:
            new.bias = self.bias.detach().to(device=device, dtype=dtype).requires_grad_()
        new._device = device
        new._dtype = dtype
        return new 
    @property
    def use_bias(self):
        return self._use_bias
    
    def clone(self):
        linear = Linear(self.d_in, self.d_out, device=self._device, dtype=self._dtype, init_method=self.init_method, use_bias=self.use_bias)
        linear.weight = self.weight.detach().clone().requires_grad_()
        if self.use_bias:
            linear.bias = self.bias.detach().clone().requires_grad_()
            
        return linear
    
    def init_like(self):
        return Linear(self.d_in, self.d_out, device=self._device, dtype=self._dtype, init_method=self.init_method, use_bias=self.use_bias)
    
    def state_dict(self):
        return {
            'weight': self.weight.detach().cpu(),
            'bias': self.bias.detach().cpu() if self.use_bias else None,
            'weight_grad': self.weight.grad.detach().cpu() if self.weight.grad is not None else None,
            'bias_grad': self.bias.grad.detach().cpu() if self.use_bias and self.bias.grad is not None else None
        }
    
    def load_state(self, state):
        super().state_key_check_with_raise(state.keys(), ['weight', 'bias', 'weight_grad', 'bias_grad'], Linear, 'load_state')
        with torch.no_grad():
            self.weight.copy_(state['weight'].to(self._device, self._dtype).requires_grad_())
            if state['weight_grad'] is not None:
                self.weight.grad = state['weight_grad'].to(self._device, self._dtype)
            
            if self.use_bias:
                self.bias.copy_(state['bias'].to(self._device, self._dtype).requires_grad_())
                if state['bias_grad'] is not None:
                    self.bias.grad = state['bias_grad'].to(self._device, self._dtype)
                    
    def __repr__(self):
        return f'Linear(d_in={self.d_in}, d_out={self.d_out}, device={self._device}, dtype={self._dtype}, use_bias={self.use_bias})'