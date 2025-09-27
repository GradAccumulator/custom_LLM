import torch
from .activation_function import ActivationFunction
from src.utils.torch_utils import 보이면_바로_교체하기

class ReLU(ActivationFunction):
    def __init__(self):
        pass
    def __call__(self, x:torch.Tensor):
        return torch.where(x>0, x, torch.tensor(0))
    
    def __repr__(self):
        return "ReLU"
    
class LeakyReLU(ActivationFunction):
    def __init__(self, a=0.1):
        self.a = a

    def forward(self, x:torch.Tensor):
        return torch.where(x>0, x, x*self.a)
    def __call__(self, x:torch.Tensor):
        return self.forward(x)
    
    def __repr__(self):
        return f"LeakyReLU(a={self.a})"

from src.modules.base_module import BaseModule
class PReLU(BaseModule, ActivationFunction):
    def __init__(self, a_init=0.1, device='cpu', dtype=torch.float32):
        self.device = device
        self._dtype = dtype
        self._a = torch.tensor(a_init, requires_grad=True, dtype=dtype, device=device)
        self.a_init = a_init

    def forward(self, x:torch.Tensor):
        return torch.where(x>0, x, x*self._a)
    def __call__(self, x:torch.Tensor):
        return self.forward(x)
    
    @property
    def a(self):
        return self._a
    def clone(self):
        prelu = PReLU(a_init=self._a.tolist(), device=self.device, dtype=self._dtype)
        prelu._a = self._a.detach().clone().requires_grad_()
        return prelu
    def init_like(self):
        return PReLU(a_init=self.a_init, device=self.device, dtype=self._dtype)
    def to(self, device=None, dtype=None):
        device, dtype = 보이면_바로_교체하기(device, dtype)
        if (self.device==device or device is None) and (self._dtype==dtype or dtype is None):
            return self
        prelu = PReLU(a_init=self.a_init, device=device or self.device, dtype=dtype or self._dtype)
        prelu._a = self._a.detach().to(device=prelu.device, dtype=prelu._dtype).requires_grad_()
        return prelu
    
    def state_dict(self):
        return {
            'a' :self._a.detach().cpu(),
            'a_grad':self._a.grad.detach().cpu()
        }
        
    def load_state(self, state):
        self.state_key_check_with_raise(state.keys(), ['a', 'a_grad'], 'load_state')
        with torch.no_grad():
            self._a.copy_(state['a'].to(device=self.device, dtype=self._dtype).requires_grad_())
            self._a.grad = state['a_grad'].to(device=self.device, dtype=self._dtype)
            
    def __repr__(self):
        return f'PReLU(a={self.a.tolist()})'