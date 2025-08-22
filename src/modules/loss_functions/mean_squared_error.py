import torch

class MeanSquaredError:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(MeanSquaredError, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        pass
    
    def mean_squared_error(self, pred:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
        return torch.mean((pred-label)**2)
    
    def __call__(self, pred:torch.Tensor, label:torch.Tensor) -> torch.Tensor:
        return self.mean_squared_error(pred, label)
