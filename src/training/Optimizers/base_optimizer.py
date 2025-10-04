import torch
from typing import (Callable, Any, NewType)
import src.c_modules as _C
from src.utils.dev_utils import alert

class BaseOptimizer:
    def __init__(self, parameters:dict[str, torch.tensor], lr:float, weight_decay:float=0, clip_grad:float|bool=True, lr_scale:dict={}):
        '''
        Args:
        parameters: a dict of parameters
        lr: learning rate  
        weight_decay: weight decay
        clip_grad: Whether to apply gradient clipping during training. or max norm. if you set clip_grad as True, the max norm is set to 1
        '''
        self.lr = lr
        self._ld = lr_scale or {}
        self._cached_params_as_dict = parameters
        self._cached_params_as_list = list(self._cached_params_as_dict.values())
        self.weight_decay = weight_decay
        if isinstance(clip_grad, bool):
            if clip_grad:
                self.max_grad_norm = 1
                self.clip_grad = True
                self.total_grad_norm = 0
                self.prev_grad_norm = 0
            else:
                self.clip_grad = False
        else:
            self.max_grad_norm = clip_grad
            self.clip_grad = True
            self.total_grad_norm = 0
            self.prev_grad_norm = 0
        

    def parameters_as_dict(self):
        return self._cached_params_as_dict
        
    def parameters(self):
        return self._cached_params_as_list
    
    @property
    def named_parameters(self):
        return self.parameters_as_dict()
    
    @property
    def weight_decay(self):
        return self._wd
    @weight_decay.setter
    def weight_decay(self, wd):
        self._wd = wd
        parameters = self.parameters_as_dict()
        if wd == 0:
            self.step = self.step_with_none
            self._params['others'] = {key:parameters[key] for key in parameters.keys() if key not in self._ld.keys()}
            self._params.update({
                "lr_scale":{
                    key:(parameters[key], self._ld[key]) 
                        for key in parameters.keys() if key in self._ld.keys()
                } 
            })
        else:
            self.step = self.step_with_wd
            self._params = {
                "weight_decay":{key:parameters[key] for key in parameters.keys() if 'weight' in key and key not in self._ld.keys()},
                'others':{key:parameters[key] for key in parameters.keys() if 'weight' not in key and key not in self._ld.keys()},
                'ls_weight_decay': {key:(parameters[key], self._ld[key]) for key in parameters.keys() if 
                    'weight' in key and key in self._ld.keys()},
                'ls_others': {key:(parameters[key], self._ld[key]) for key in parameters.keys() if 
                    'weight' not in key and key in self._ld.keys()}
            }
    
    FalseTensor = NewType('FalseTensor', torch.Tensor)
    @torch.no_grad()
    def clip_grad_norm(self)->torch.Tensor|FalseTensor:
        if self.clip_grad:
            return torch.nn.utils.clip_grad_norm_(self.named_parameters(), max_norm=self.max_grad_norm)
        return torch.tensor(False)

    @torch.no_grad()
    def step_with_wd(self, compute_grad_fn:Callable[...,dict[str,torch.Tensor]], *args, **kwargs):
        self.prev_grad_norm = self.clip_grad_norm()
        gradients = compute_grad_fn(*args, **kwargs)
        try:
            _C.step_with_wd(
                self._params,
                gradients,
                self.lr,
                self.weight_decay
            )
        except Exception as e:
            print(f'error in optimizer step_with_wd')
            breakpoint()
            raise e
    
    @torch.no_grad()
    def step_with_none(self, compute_grad_fn:Callable[...,dict[str,torch.Tensor]], *args, **kwargs):
        self.prev_grad_norm = self.clip_grad_norm()
        gradients = compute_grad_fn(*args, **kwargs)
        _C.step_with_none(
            self._params,
            gradients,
            self.lr
        )
    
    @torch.no_grad()
    def zero_grad(self, set_to_none=False):
        _C.zero_grad(self.named_parameters(), set_to_none=set_to_none)

    def parameters_append(self, **kwargs):
        if self._wd != 0:
            for k,v in kwargs.items():
                if "weight" in k:
                    self._params['weight_decay'][k] = v
                else:
                    self._params['others'][k] = v
        else:
            self._params.update(kwargs)
        self._cached_params_as_dict.update(kwargs)
        self._cached_params_as_list.extend(list(kwargs.values()))

    def state_dict(self):
        state = {
            'lr':self.lr,
            'wd':self._wd
        }
        return state