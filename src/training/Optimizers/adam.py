import torch
from .base_optimizer import BaseOptimizer

class Adam(BaseOptimizer):
    @staticmethod
    def load_state(params:dict, state_dict:dict):
        adam = Adam(
            params,
            lr=state_dict["lr"],
            beta1=state_dict['b1'],
            beta2=state_dict['b2'],
            weight_decay=state_dict['wd'],
            eps=state_dict['eps'],
            skip_init=True
        )
        adam.m = state_dict['m']
        adam.v = state_dict['v']
        adam.t = state_dict['t']
        for k in adam.m.keys():
            adam.m[k] = adam.m[k].to(adam._cached_params_as_dict[k].device)
            adam.v[k] = adam.v[k].to(adam._cached_params_as_dict[k].device)
        return adam

    def __init__(self, parameters:dict, lr:float, beta1:float, beta2:float, weight_decay:float=0,*, eps=1e-8, clip_grad:float|bool=True, skip_init = False, lr_scale:dict=None):
        super().__init__(parameters=parameters, lr=lr,weight_decay=weight_decay, lr_scale=lr_scale, clip_grad=clip_grad)
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        
        if not skip_init:
            self.m = {key: torch.zeros_like(parameters[key], requires_grad=False) for key in parameters.keys()}
            self.v = {key: torch.zeros_like(parameters[key], requires_grad=False) for key in parameters.keys()}

    def update_mv(self) -> dict[str, torch.Tensor]:
        self.t += 1
        params = self.parameters_as_dict()
        adam_updated = {}
        for key in params.keys():
            try:
                self.m[key] = self.b1*self.m[key] + (1-self.b1)*params[key].grad
                self.v[key] = self.b2*self.v[key] + (1-self.b2)*(params[key].grad**2)

                m_hat = self.m[key]/(1-self.b1**self.t)
                v_hat = self.v[key]/(1-self.b2**self.t)

                adam_updated[key] = m_hat/(v_hat.sqrt()+self.eps)
            except Exception as e:
                print(f"error in adam update in {key}")
                raise e
        return adam_updated

    def step_with_wd(self):
        with torch.no_grad():
            super().step_with_wd(self.update_mv)

    def step_with_none(self):
        with torch.no_grad():
            super().step_with_none(self.update_mv)

    def parameters_append(self, **kargs):
        super().parameters_append(**kargs)
        for key in kargs.keys():
            self.m[key] = torch.zeros_like(self._params[key], requires_grad=False)
            self.v[key] = torch.zeros_like(self._params[key], requires_grad=False)

    def state_dict(self):
        m = {}
        v = {}
        for key in self.parameters_as_dict().keys():
            m[key] = self.m[key].cpu().detach().clone()
            v[key] = self.v[key].cpu().detach().clone()
        state = super().state_dict()
        state['b1'] = self.b1
        state['b2'] = self.b2
        state['eps'] = self.eps
        state['m'] = m
        state['v'] = v
        state['t'] = self.t
        return state
    
    def __repr__(self):
        return (
            f"Adam(\n"
            f'    parameter keys={list(self.parameters_as_dict().keys())},\n'
            f'    lr={self.lr},\n'
            f'    beta1={self.b1},\n'
            f'    beta2={self.b2},\n'
            f'    weight_decay={self.weight_decay},\n'
            f'    eps={self.eps},\n'
            f'    clip_grad={self.clip_grad}\n'
            f')'
        )
