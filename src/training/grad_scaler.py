import torch
import src.c_modules as _C

class GradScaler:
    def __init__(
        self,
        init_scale=2**12,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    ):
        self.scalar = torch.amp.GradScaler(init_loss_scale=init_scale)
        self.init_scale = init_scale
        self.scalar = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._enabled = enabled
        self.not_finite_count = 0
        self._growth_tracker = 0
    
    def scale(self, value:torch.Tensor):
        if self._enabled:
            return value*self.scalar
        else:
            return value
    
    def _unscale(self, optimizer):
        if not self._enabled:
            return
        _C.unscale_grad(optimizer.parameters(), self.scalar)
    
    def step(self, optimizer):
        if not self._enabled:
            return optimizer.step()
        if _C.grad_nan_check(optimizer.parameters()):
            self.not_finite_count += 1
            self.scalar*=self.backoff_factor
            return
        else:
            self._growth_tracker+=1
            if self._growth_tracker%self.growth_interval==0:
                self.scalar*=self.growth_factor
        self._unscale(optimizer)
        optimizer.step()
    
    def update(self):
        