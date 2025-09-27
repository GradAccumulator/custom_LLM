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
        super().__init__(device, dtype, d_in=d_in, d_out=d_out, device=device, dtype=dtype, init_method=init_method, use_bias=use_bias)
        self.init_method = init_method
        self.d_in = d_in
        self.d_out = d_out
        self._use_bias = use_bias
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
    @property
    def use_bias(self):
        return self._use_bias
    
    def __repr__(self):
        return f'Linear(d_in={self.d_in}, d_out={self.d_out}, device={self._device}, dtype={self._dtype}, use_bias={self.use_bias})'