import torch
from typing import (Callable, Any, ParamSpecKwargs)
import src.c_modules as _C
from src.utils.dev_utils import alert

class BaseOptimizer:
    def __init__(self, parameters:dict[str, torch.tensor], lr:float, weight_decay:float=0, clip_grad:float|bool=True, lr_scale:dict={}, loss_scale:float=1):
        '''
        parameters: a dict of parameters                                                    \n
        lr: learning rate                                                                   \n                          
        weight_decay: weight decay                                                          \n                  
        clip_grad: Whether to apply gradient clipping during training. or max norm. if you set clip_grad as True, the max norm is set to 1
        '''
        self.lr = lr
        self._ld = lr_scale or {}
        self._cached_params_as_dict = parameters
        self._cached_params_as_list = list(self._cached_params_as_dict.values())
        self.weight_decay = weight_decay
        self.recip_loss_scale = 1/loss_scale
        self.check_nan = True
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
    def params(self):
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
    
    def clip_grad_norm(self, grad_accum=1):
        if self.clip_grad:
            with torch.no_grad():
                return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm/self.recip_loss_scale*grad_accum)*self.recip_loss_scale/grad_accum
        return torch.tensor(False)

    def step_with_wd(self, compute_grad_fn:Callable[[Any],dict[str,torch.Tensor]], grad_accum=1, **kwargs):
        with torch.no_grad():
            self.prev_grad_norm = self.clip_grad_norm(grad_accum)
            gradients = compute_grad_fn(**kwargs)
            try:
                _C.step_with_wd(
                    self._params,
                    gradients,
                    self.lr,
                    self.weight_decay
                )
                # for k, p in self._params['weight_decay'].items():
                #     p.sub_(self.lr * (gradients[k]  + self._wd*p))
                # for k,p in self._params['others'].items():
                #     p.sub_(self.lr  * gradients[k])
                # for k,(p,ld) in self._params['ls_weight_decay'].items():
                #     p.sub_(self.lr * ld*( gradients[k] + self._wd*p))
                # for k,(p,ld) in self._params['ls_others'].items():
                #     p.sub_(self.lr  * ld*gradients[k])
            except Exception as e:
                print(f'error in optimizer step_with_wd')
                breakpoint()
                raise e
    def step_with_none(self, compute_grad_fn:Callable[[],dict[str,torch.Tensor]], grad_accum=1, **kwargs):
        with torch.no_grad():
            self.prev_grad_norm = self.clip_grad_norm(grad_accum)
            gradients = compute_grad_fn(**kwargs)
            _C.step_with_none(
                self._params,
                gradients,
                self.lr
            )
                
    def zero_grad(self):
        with torch.no_grad():
            _C.zero_grad(self.parameters())

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