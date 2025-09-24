import torch
from .activation_function import ActivationFunction
import src.c_modules as _C

class Tanh(ActivationFunction):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Tanh, cls).__new__(cls)
        return cls.instance
    def __init__(self):...
        
    def forward(self, x:torch.Tensor):
        return _C.tanh(x)
    
    def __call__(self, x:torch.Tensor):
        return self.forward(x)
    
    def __repr__(self):
        return "Tanh()"