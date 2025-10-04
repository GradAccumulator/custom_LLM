import torch
from .base_module import Module
from typing import Optional, Callable, overload, Any

class _linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx:torch.autograd.function.FunctionCtx, input, weight, bias=None):
        ctx.save_for_backward(
            input   , 
            weight  , 
        )
        ctx.bias_is_none = bias is None
        if ctx.bias_is_none:
            return input @ weight.T.to(input.dtype)
        return input @ weight.T.to(input.dtype) + bias.to(input.dtype)
    @staticmethod
    def backward(ctx:torch.autograd.function.FunctionCtx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight.to(grad_output.dtype)
        grad_weight = grad_output.transpose(-2,-1) @ input.to(grad_output.dtype)
        if not ctx.bias_is_none:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias

class Linear(Module):
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
        else:
            self.bias = None
    
    def forward(self, x:torch.Tensor):
        return _linear.apply(x, self.weight, self.bias)
    @property
    def use_bias(self):
        return self._use_bias
    
    def __repr__(self):
        return f'Linear(d_in={self.d_in}, d_out={self.d_out}, device={self._device}, dtype={self._dtype}, use_bias={self.use_bias})'