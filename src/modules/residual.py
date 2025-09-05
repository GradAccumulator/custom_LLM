from torch import Tensor
class Residual:
    def __init__(self):
        self.value = None

    def save_wrapper(self, value):
        self.value = value
        return value
    def sum_wrapper(self, value):
        saved = self.value
        self.value = None
        return saved + value

    def save(self):
        return self.save_wrapper
    def sum(self):
        return self.sum_wrapper
    
    def clone(self):
        res = Residual()
        res.value = self.value.detach().clone() if isinstance(self.value, Tensor) else self.value
        return res
    def init_like(self):
        return Residual()
    
    def __repr__(self):
        return f'Residual(value={self.value})'