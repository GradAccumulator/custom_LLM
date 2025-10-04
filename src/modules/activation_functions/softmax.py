import torch
from .activation_function import ActivationFunction

class Softmax(ActivationFunction):
    def __init__(self, dim=-1, temperature=1.0):
        self.dim=dim
        self.reciprocal_temp = 1/temperature
        self.temp = temperature
        if temperature==1.0:
            self._exp = self.exp_without_temp
        else:
            self._exp = self.exp_with_temp

    def exp_without_temp(self, x:torch.Tensor):
        return x.exp()
    def exp_with_temp(self, x:torch.Tensor):
        return (x*self.reciprocal_temp).exp()

    def forward(self, x:torch.Tensor):
        scaled = x - x.max(dim=self.dim, keepdim=True).values
        exps = self._exp(scaled)
        sum = exps.sum(dim=self.dim, keepdim=True)
        return exps/sum
    
    def __repr__(self):
        return f"Softmax(dim={self.dim}, temperature={self.temp})"