import torch

class Reshape:
    def __init__(self, *shape):
        self.shape = shape
    
    def forward(self, x:torch.Tensor):
        return x.reshape(self.shape)
    
    def __call__(self, x:torch.Tensor):
        return self.forward(x)