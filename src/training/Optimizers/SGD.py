import torch
from .base_optimizer import BaseOptimizer

class SGD(BaseOptimizer):
    def __init__(self, parameters, lr, weight_decay = 0, momentum = 0, nesterov = False, dampening = 0, **lr_decay):
        super().__init__(parameters, lr, weight_decay, lr_decay=lr_decay)
        self._momentum = momentum
        self._nesterov = nesterov
        self._dampening = dampening
        self.momentum = momentum
        
        # if self._momentum != 0:
        #     self.v:dict[str,torch.Tensor] = {key: torch.zeros_like(params, requires_grad=False) for key, params in self.parameters.items()}
        #     if nesterov:
        #         self.compute_gradients_fn = self.gradients_nesterov
        #     else:
        #         self.compute_gradients_fn = self.gradients_momentum
            
        #     if self._dampening != 0:
        #         self.update_velocity = self.update_velocity_with_dampening
        #     else:
        #         self.update_velocity = self.update_velocity_without_dampening
        # else:
        #     self.compute_gradients_fn = self.gradients_SGD

    @property
    def momentum(self):
        return self._momentum
    @momentum.setter
    def momentum(self, momentum):
        self._momentum = momentum
        if self._momentum != 0:
            self.v = {key: torch.zeros_like(params, requires_grad=False) for key, params in self.parameters.items()}
            if self.nesterov:
                self.compute_gradients_fn = self.gradients_nesterov
            else:
                self.compute_gradients_fn = self.gradients_momentum
            
            if self.dampening != 0:
                self.update_velocity = self.update_velocity_with_dampening
            else:
                self.update_velocity = self.update_velocity_without_dampening
        else:
            self.compute_gradients_fn = self.gradients_SGD

    @property
    def nesterov(self):
        return self._nesterov
    @nesterov.setter
    def nesterov(self, nesterov):
        self._nesterov = nesterov
        if self._momentum != 0:
            if self._nesterov:
                self.compute_gradients_fn = self.gradients_nesterov
            else:
                self.compute_gradients_fn = self.gradients_momentum
        else:
            self.compute_gradients_fn = self.gradients_SGD

    @property
    def dampening(self):
        return self._dampening
    @dampening.setter
    def dampening(self, dampening):
        self._dampening = dampening
        if self._momentum != 0:
            if self._dampening != 0:
                self.update_velocity = self.update_velocity_with_dampening
            else:
                self.update_velocity = self.update_velocity_without_dampening

    def update_velocity_with_dampening(self):
        for k,p, in self.parameters_dict().items():
            self.v[k] = self._momentum*self.v[k] + self._dampening*p.grad
    def update_velocity_without_dampening(self):
        for k,p, in self.parameters_dict().items():
            self.v[k] = self._momentum*self.v[k]+p.grad
    
    def gradients_nesterov(self) -> dict[str,torch.Tensor]:
        with torch.no_grad():
            self.update_velocity()
            gradients ={}
            for k,p in self.parameters_dict().items():
                gradients[k] = p.grad + self._momentum*self.v[k]
        return gradients
    
    def gradients_momentum(self)-> dict[str,torch.Tensor]:
        with torch.no_grad():
            self.update_velocity()
            return self.v
        
    def gradients_SGD(self):
        return self.parameters_dict()

    def step_with_wd(self):
        super().step_with_wd(self.compute_gradients_fn)
    
    def step_with_none(self):
        super().step_with_none(self.compute_gradients_fn)

    def state_dict(self):
        state = super().state_dict()
        state['momentum'] = self._momentum
        state['nesterov'] = self._nesterov
        state['dampening'] = self._dampening
        if self._momentum != 0:
            state['v'] = self.v
        return state
    
    @staticmethod
    def load_state(state):
        sgd = SGD(state['parameters'], state['lr'], state['weight_decay'], state['momentum'], state['nesterov'], state['dampening'])
        if state['momentum'] != 0:
            sgd.v = state['v']
        return sgd
