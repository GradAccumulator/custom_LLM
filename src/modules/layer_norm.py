import torch
from .base_module import BaseModule
from src.utils.torch_utils import device_dtype_check

class LayerNorm(BaseModule):
    def __init__(self, dim,*,eps=5e-5, device='cpu', dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.eps = eps
        self.dim = dim
        self.gamma = torch.ones(self.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        self.beta = torch.zeros(self.dim, device=self.device, dtype=self.dtype, requires_grad=True)

    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x-mean)/(var+self.eps).sqrt()
        out = self.gamma.to(x.dtype)*out + self.beta.to(x.dtype)
        return out
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device=None, dtype=None):
        device, dtype = device_dtype_check(device, dtype)
        if (self.device==device or device is None) and (self.dtype==dtype or dtype is None):
            return self
        ln = self.clone()
        ln.device = device if device is not None else self.device
        ln.dtype = dtype if dtype is not None else self.dtype
        ln.gamma = self.gamma.detach().to(device=ln.device, dtype=ln.dtype).requires_grad_()
        ln.beta = self.beta.detach().to(device=ln.device, dtype=ln.dtype).requires_grad_()
        return ln
        
    def clone(self):
        ln = LayerNorm(dim=self.dim, eps=self.eps, device=self.device, dtype=self.dtype)
        ln.gamma = self.gamma.detach().clone().requires_grad_()
        ln.beta = self.beta.detach().clone().requires_grad_()
        return ln
    
    def init_like(self):
        return LayerNorm(dim=self.dim, eps=self.eps, device=self.device, dtype=self.dtype)
    
    def state_dict(self):
        return {
            'gamma':self.gamma.detach().cpu(),
            'beta':self.beta.detach().cpu(),
            'gamma_grad':self.gamma.grad.detach().cpu() if self.gamma.grad is not None else None,
            'beta_grad':self.beta.grad.detach().cpu() if self.beta.grad is not None else None
        }
        
    def load_state(self, state):
        self.state_key_check_with_raise(state.keys(), ['gamma', 'beta', 'gamma_grad', 'beta_grad'], LayerNorm, 'load_state')
        with torch.no_grad():
            self.gamma.copy_(state['gamma'].to(self.device, self.dtype).requires_grad_())
            self.beta.copy_(state['beta'].to(self.device, self.dtype).requires_grad_())
            if state['gamma_grad'] is not None:
                self.gamma.grad = state['gamma_grad'].to(self.device, self.dtype)
            if state['beta_grad'] is not None:
                self.beta.grad = state['beta_grad'].to(self.device, self.dtype)
                
    def __repr__(self):
        return f'LayerNorm(dim={self.dim}, eps={self.eps})'