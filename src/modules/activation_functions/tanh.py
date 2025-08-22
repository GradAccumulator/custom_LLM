import torch


class Tanh:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
    def forward(self, x:torch.Tensor):
        return (x.exp() - (-x).exp())/(x.exp() + (-x).exp())
    
    def __call__(self, x:torch.Tensor):
        return self.forward(x)