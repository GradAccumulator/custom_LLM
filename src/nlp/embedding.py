import torch
import math

class Embedding:
    def __init__(self, d_in, d_out, dtype=torch.float32, device='cpu'):
        self._d_in = d_in
        self._d_out = d_out
        self._dtype = dtype
        self._std_weights = math.sqrt(2/(d_in+d_out))
        self.weight = (torch.randn((d_in, d_out), dtype=dtype, device=device)*self._std_weights).requires_grad_()

    def __call__(self, x):
        return self.weight[x]
    
    @property
    def d_in(self):
        return self._d_in
    
    @property
    def d_out(self):
        return self._d_out
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        return self.weight.device
    
    def state_dict(self):
        return {
            'd_in':self.d_in,
            'd_out':self.d_out,
            'dtype':self.dtype,
            'device':self.device,
            'weights':self.weight.cpu().detach().clone()
        }
    @staticmethod
    def load_state(state:dict):
        embd = Embedding(state['d_in'], state['d_out'], state['dtype'], state['device'])
        embd.weight = state['weights'].detach().to(state['device']).requires_grad_()
        return embd