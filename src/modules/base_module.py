import torch
import math

class BaseModule(object):
    '''
    모듈에 있어야 하는거\n
    to() -> type(self) 깊은복사\n
    clone() -> type(self) 깊은복사\n
    init_like() -> type(self) 깊은복사\n
    parameters() -> list\n
    
    to는 init_like() + 파라미터들.detach().clone().requires_grad_()
    clone은 자기 타입(바뀐 device,dtype) + 파라미터들.detach().clone().requires_grad_()
    '''
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype  = dtype
        self._parameters = {}
        
    def state_key_check_with_raise(self, state_keys, keys, cls,func_name):
        '''state_keys에서 ㅏkeys에 없는게 있으면 raise 함'''
        for key in keys:
            if key not in state_keys:
                raise ValueError(
                    f'<<{cls.__name__}.{func_name}>>\n'
                    f'key({key}) is not in state dict.'
                )
        if set(state_keys) != set(keys):
            raise ValueError(
                f'<<{cls.__name__}.{func_name}>>\n'
                f'keys({keys}) and state dict({state_keys}) are not same.'
            )
    
    def init_parameter(self, d_in, d_out,p,*, init_method:str='none'):
        init_type = init_method.lower()
        with torch.no_grad():
            if init_type == 'xavier':
                p.mul_(math.sqrt(2/(d_in+d_out)))
            elif init_type == 'he':
                p.mul_(math.sqrt(2/d_in))
            elif init_type == 'none':
                pass
            else:
                raise ValueError(f'Invalid weight initialization: {init_type}')
            
    def __setattr__(self, name, value):
        if isinstance(value, torch.Tensor):
            self._parameters[name] = value
        elif isinstance(value, BaseModule):
            for k,v in value._parameters.items():
                self._parameters[f'{name}.{k}'] = v
        super().__setattr__(name, value)
        
    
    def parameters(self):
        return list(self._parameters.values())
    
    def parameters_as_dict(self):
        return self._parameters