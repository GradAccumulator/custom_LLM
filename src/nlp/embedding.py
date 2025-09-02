import torch
import math
from src.modules.base_module import BaseModule
from src.utils.torch_utils import device_dtype_check

class Embedding(BaseModule):
    def __init__(self, d_in, d_out, dtype=torch.float32, device='cpu'):
        super().__init__(device=device, dtype=dtype)
        self._d_in = d_in
        self._d_out = d_out
        self.dtype = dtype
        self._std_weights = math.sqrt(2/(d_in+d_out))
        self.table = (torch.randn((d_in, d_out), dtype=dtype, device=device)*self._std_weights).requires_grad_()


    def forward(self, x:torch.Tensor):
        return self.table[x]
    
    def __call__(self, x:torch.Tensor):
        return self.forward(x)
    
    @property
    def d_in(self):
        return self._d_in
    
    @property
    def d_out(self):
        return self._d_out
    
    @property
    def device(self):
        return self.table.device
    
    def to(self, device=None, dtype=None):
        device, dtype = device_dtype_check(device, dtype)
        device = device or self.device
        dtype = dtype or self.dtype
        embd = self.init_like()
        embd.table = self.table.detach().to(device, dtype).requires_grad_()
        embd.device = device
        embd.dtype = dtype
        return embd
    
    def clone(self):
        embd = self.init_like()
        embd.table = self.table.detach().clone().requires_grad_()
        return embd
    
    def init_like(self):
        return Embedding(self.d_in, self.d_out, device=self.device, dtype=self.dtype)
    
    def state_dict(self):
        return {
            'd_in':self.d_in,
            'd_out':self.d_out,
            'dtype':self.dtype,
            'device':self.device,
            'table':self.table.cpu().detach().clone(),
            'grad':self.table.grad.cpu().detach().clone() if self.table.grad is not None else None
        }
    @staticmethod
    def load_state(state:dict):
        embd = Embedding(state['d_in'], state['d_out'], state['dtype'], state['device'])
        embd.table = state['table'].detach().to(state['device']).requires_grad_()
        if state['grad'] is not None:
            embd.table.grad = state['grad'].detach().to(state['device'])    
        return embd