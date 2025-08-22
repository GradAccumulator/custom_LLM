import torch

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        
    def forward(self, x):
        mask = (torch.rand_like(x, dtype=torch.float16) >= self.p)
        return x * mask
        
    def __call__(self, x):
        return self.forward(x)