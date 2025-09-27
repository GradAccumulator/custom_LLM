from torch import Tensor
from .base_module import BaseModule
class Residual:
    def __init__(self):
        self.value = None

    def save_wrapper(self, value):
        self.value = value
        return value
    def sum_wrapper(self, value):
        saved = self.value
        self.value = None
        if isinstance(saved, Tensor):
            return saved.to(value.dtype) + value
        return saved + value

    def save(self):
        return self.save_wrapper
    def sum(self):
        return self.sum_wrapper
    
    def __repr__(self):
        return f'Residual()'