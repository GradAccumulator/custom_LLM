import torch
from typing import overload, Any


class Layers:
    @overload
    def __init__(self, modules:dict[str, Any], num_layers, device='cpu', dtype=torch.float32):...
    @overload
    def __init__(self, *modules, num_layers, device='cpu', dtype=torch.float32):...
    @overload
    def __init__(self, *, num_layers, device='cpu', dtype=torch.float32, **modules):...
    def __init__(self, *modules_list:dict[str]|Any, num_layers, device='cpu', dtype=torch.float32, **modules_dict):
        if modules_dict and modules_list:
            raise ValueError('modules_dict와 modules_list를 동시에 전달할 수 없습니다.')
        elif not modules_dict and not modules_list:
            raise ValueError('modules_dict와 modules_list가 모두 비어있습니다.')
        elif isinstance(modules_list[0], dict):
            self.is_dict = True
            modules = modules_list[0]
            if len(modules_list) > 1:
                raise ValueError('modules를 딕셔너리로 전달할 때는 여러개의 인자를 전달할 수 없습니다.')
        elif modules_dict:
            modules = modules_dict
            self.is_dict = True
        else:
            modules = dict(enumerate(modules_list))
            self.is_dict = False
        
        self.modules = modules
        self.dtype = dtype
        self.device = device
        self.num_layers = num_layers
        self._set_layers(modules)

    def _set_layers(self, modules:dict):
        self.layers = [{} for _ in range(self.num_layers)]
        for i in range(self.num_layers):
            for k,v in modules.items():
                if hasattr(v, 'init_like'):
                    self.layers[i][k] = v.init_like().to(device=self.device, dtype=self.dtype)
                else:
                    self.layers[i][k] = v.to(device=self.device, dtype=self.dtype)
        self.cache_parameters()
    
    def cache_parameters(self):
        self.params_as_dict = {}
        for k,m in self.modules.items():
            if hasattr(m, 'parameters_as_dict'):
                self.params_as_dict[k] = m.params_as_dict()
            elif hasattr(m, 'parameters'):
                self.params_as_dict[k] = m.parameters()
        self.params_as_list = [p for module in self.modules.values() if hasattr(module, 'parameters') for p in module.parameters()]

    def parameters_as_dict(self):
        return self.params_as_dict if hasattr(self, 'params_as_dict') else None
    def parameters(self):
        return self.params_as_list
    
    def forward(self, x):
        for layer in self.layers:
            for module in layer.values():
                x = module(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def __getitem__(self, idx:int) -> dict[str,Any]:
        if idx>=len(self.layers):
            raise IndexError
        return self.layers[idx]
    
    def to(self, device=None, dtype=None):
        if (self.device==device or device is None) and (self.dtype == dtype or dtype is None):
            return self
        new = Layers(self.modules, self.num_layers, self.device, self.dtype)
        for i in range(self.num_layers):
            for k,m in self.modules.items():
                if hasattr(m, 'to'):
                    new[i][k] = m.to(device=device if device else self.device, dtype=dtype if dtype else self.dtype)
                else:
                    new[i][k] = m
        return new
    
    def state_dict(self):
        return {i:{k:v.state_dict() for k,v in layer.items() if hasattr(v, 'state_dict')} for i,layer in enumerate(self.layers)}
    
    def load_state(self, state):
        for i,layer in state.items():
            for k,v in layer.items():
                if not hasattr(self[i][k], 'load_state'):
                    raise ValueError(
                        '<<Layers.load_state>>\n'
                    f"Layer {i}, Module {k} has not load_state method.\n"
                    "Cannot load state for this layer.\n"
                    f'(layer keys: {list(layer.keys())})'
                )
                self[i][k].load_state(v)