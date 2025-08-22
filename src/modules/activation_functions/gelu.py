import torch
import math
from .tanh import Tanh

class GELU:
    def __init__(self, erf=True):
        if erf:
            self.forward = self.gelu_erf
            self.sqrt_2 = math.sqrt(2)
        else:
            self.sqrt_2_over_pi = math.sqrt(2/math.pi)
            self.forward = self.gelu_tanh
            self.tanh = Tanh()

    def gelu_erf(self, x:torch.Tensor):
        return 0.5*x*(1+torch.erf(x/self.sqrt_2))
    
    def gelu_tanh(self, x:torch.Tensor):
        return 0.5*x*(1+self.tanh(self.sqrt_2_over_pi*(x+0.044715*(x**3))))
    
    def __call__(self, x:torch.Tensor):
        return self.forward(x)
