import torch
from .base_module import Module

class _layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx:torch.autograd.function.FunctionCtx, x, gamma, beta, eps):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x-mean)/(eps + std)
        out = gamma.to(x.dtype) * x_hat + beta.to(x.dtype)
        ctx.eps = eps
        ctx.save_for_backward(
            x       ,
            gamma   ,
            beta    ,
            x_hat   ,
            mean    ,
            std     ,
        )
        return out
    
    @staticmethod
    def backward(ctx:torch.autograd.function.FunctionCtx, grad_output):
        x,gamma,beta,x_hat,mean,std = ctx.saved_tensors
        
        x       = x.to(grad_output.dtype)
        gamma   = gamma.to(grad_output.dtype)
        beta    = beta.to(grad_output.dtype)
        x_hat   = x_hat.to(grad_output.dtype)
        mean    = mean.to(grad_output.dtype)
        std     = std.to(grad_output.dtype)
        
        
        u = grad_output*gamma
        gamma_grad = (grad_output*x_hat).sum(dim=-1, keepdim=False)
        beta_grad  = grad_output.sum(dim=-1, keepdim=False)
        
        grad_output.mul_(gamma)
        m1 = u.mean(dim=-1, keepdim=True)
        m2 = (u*x_hat).mean(dim=-1, keepdim=True)
        
        x_grad = (1/(std+ctx.eps))*(u-m1-x_hat*m2)
        
        return x_grad, gamma_grad, beta_grad, None

class LayerNorm(Module):
    def __init__(self, dim,*,eps=5e-5, device='cpu', dtype=torch.float32):
        super().__init__(device, dtype, dim=dim, eps=eps, device=device, dtype=dtype)
        self.eps = eps
        self.dim = dim
        self.gamma = torch.ones(self.dim, device=self.device, dtype=self.dtype, requires_grad=True)
        self.beta = torch.zeros(self.dim, device=self.device, dtype=self.dtype, requires_grad=True)

    def forward(self, x:torch.Tensor):
        return _layernorm.apply(x, self.gamma, self.beta, self.eps)
        
    def __repr__(self):
        return f'LayerNorm(dim={self.dim}, eps={self.eps})'