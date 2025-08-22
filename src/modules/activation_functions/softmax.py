import torch

class Softmax:
    def __init__(self, dim=-1, dtype=torch.float32, temperature=1.0):
        self.dim=dim
        self.reciprocal_temp = 1/temperature
        self.temp = temperature
        if temperature==1.0:
            self._exp = self.exp_without_temp
        else:
            self._exp = self.exp_with_temp

        if dtype==torch.float32 or dtype==torch.float64:
            self.softmax = self.softmax_fp32_or_over
        else:
            self.dtype = dtype
            self.softmax = self.softmax_fp32_under

    def exp_without_temp(self, x:torch.Tensor):
        return x.exp()
    def exp_with_temp(self, x:torch.Tensor):
        return (x*self.reciprocal_temp).exp()

    def softmax_fp32_or_over(self, x:torch.Tensor):
        scaled = x - x.max(dim=self.dim, keepdim=True).values
        exps = self._exp(scaled)
        sum = exps.sum(dim=self.dim, keepdim=True)
        return exps/sum
    def softmax_fp32_under(self, x:torch.Tensor):
        scaled = x - x.max(dim=self.dim, keepdim=True).values
        exps = self._exp(scaled)
        sum = exps.sum(dim=self.dim, keepdim=True)
        return (exps/sum)
    
    def to(self, device=None, dtype=None):
        if self.dtype==dtype or dtype is None:
            return self
        return Softmax(dim=self.dim, dtype=dtype, temperature=self.temp)
    
    def __call__(self, x:torch.Tensor):
        return self.softmax(x)