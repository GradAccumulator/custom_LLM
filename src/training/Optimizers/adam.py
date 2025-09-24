import torch
from .base_optimizer import BaseOptimizer
import src.c_modules as _C

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

    def __init__(self, parameters:dict, lr:float, beta1:float, beta2:float, weight_decay:float=0,*, eps=1e-8, clip_grad:float|bool=True, skip_init = False, lr_scale:dict=None, device_transfer_while_conputing_mv = False, loss_scale=1.0):
        '''
        device_transfer_while_conputing_mv: True일때는 Adam.step()을 호출 한 뒤에 m,v를 수정할 때만 gpu로 옮겨서 작업하고, 이후 다시 CPU로 옮겨놓음. grad accumulation을 큰 값으로 사용할 때 사용
        '''
        super().__init__(parameters=parameters, lr=lr,weight_decay=weight_decay, lr_scale=lr_scale, clip_grad=clip_grad, loss_scale=loss_scale)
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.use_device_transfer = device_transfer_while_conputing_mv
        self.t = 0
        self.device = next(iter(parameters.values())).device
        
        if not skip_init:
            self.m = {key: torch.zeros_like(parameters[key], requires_grad=False, device = self.device if not self.use_device_transfer else 'cpu') for key in parameters.keys()}
            self.v = {key: torch.zeros_like(parameters[key], requires_grad=False, device = self.device if not self.use_device_transfer else 'cpu') for key in parameters.keys()}
            
    def update_mv(self, **kwargs) -> dict[str, torch.Tensor]:
        self.t += 1
        params = self.parameters_as_dict()
        try:
            adam_updated = _C.computeMV_wrapper(
                params,
                self.m,
                self.v,
                self.t,
                self.b1,
                self.b2,
                self.eps,
                self.use_device_transfer
            )
        except Exception as e:
            print(f"error in adam update")
            raise e
        return adam_updated

    def step_with_wd(self, grad_accum=1, **kwargs):
        with torch.no_grad():
            super().step_with_wd(self.update_mv, grad_accum=grad_accum, **kwargs)

    def step_with_none(self, grad_accum=1, **kwargs):
        with torch.no_grad():
            super().step_with_none(self.update_mv, grad_accum=grad_accum, **kwargs)

    def parameters_append(self, **kargs):
        super().parameters_append(**kargs)
        for key in kargs.keys():
            self.m[key] = torch.zeros_like(self._params[key], requires_grad=False, device=self.device if not self.use_device_transfer else 'cpu')
            self.v[key] = torch.zeros_like(self._params[key], requires_grad=False, device=self.device if not self.use_device_transfer else 'cpu')

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
            f'    lr={self.lr},\n'
            f'    beta1={self.b1},\n'
            f'    beta2={self.b2},\n'
            f'    weight_decay={self.weight_decay},\n'
            f'    eps={self.eps},\n'
            f'    clip_grad={self.clip_grad}\n'
            f')'
        )
