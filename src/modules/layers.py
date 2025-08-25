import torch
from typing import overload, Any
from src.utils.torch_utils import device_dtype_check
from src.modules.to import To
from .base_module import BaseModule


class LayerWrapper(dict):
    def __init__(self, layer:dict[str, Any], idx:int):
        super().__init__(layer)
        self.idx = idx
    def __repr__(self):
        return (
            f'Layers-LayerWrapper(idx={self.idx})(\n'
            f'        <<modules>>\n        '+
            f'\n        '.join(f'[{k}]:{v}' for k,v in self.items())+
            f'\n)'
        )
    
    def __getitem__(self, key):
        return super().__getitem__(str(key))
    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)
    
    @staticmethod
    def wrap_layers(layers:list[dict[str,Any]]):
        for i in range(len(layers)):
            layers[i] = LayerWrapper(layers[i], i)
        return layers

class Layers(BaseModule):
    @overload
    def __init__(self, modules:dict[str, Any], num_layers, device='cpu', dtype=torch.float32):...
    @overload
    def __init__(self, *modules, num_layers, device='cpu', dtype=torch.float32):...
    @overload
    def __init__(self, *, num_layers, device='cpu', dtype=torch.float32, **modules):...
    def __init__(self, *modules_list, num_layers, device='cpu', dtype=torch.float32, **modules_dict):
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
            modules = {}
            for i,m in enumerate(modules_list):
                modules[str(i)] = m
            self.is_dict = False
        
        super().__init__(device=device, dtype=dtype)
        self.modules:dict[str, Any] = modules
        self.num_layers = num_layers
        self._set_layers(modules)

    def _set_layers(self, modules:dict):
        self.layers:list[LayerWrapper] = [{} for _ in range(self.num_layers)]
        for i in range(self.num_layers):
            for k,m in modules.items():
                if hasattr(m, 'init_like'):
                    m = m.init_like()
                if not isinstance(m, To) and hasattr(m, 'to'):
                    self.layers[i][k] = m.to(device=self.device, dtype=self.dtype)
                else:
                    self.layers[i][k] = m
            self.set_layer(i, self.layers[i])
    
    def forward(self, x):
        for layer in self.layers:
            for module in layer.values():
                x = module(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def __getitem__(self, idx:int|str) -> dict[str,Any]:
        if idx>=len(self.layers):
            raise IndexError
        return self.layers[idx]
        
    def set_module(self, idx:int, key:str, module):
        if idx>=len(self.layers):
            raise IndexError
        self.layers[idx][key] = module
        if isinstance(module, BaseModule):
            for k,p in module._parameters.items():
                self._parameters[f'layers[{idx}][{key}]({self.layers[idx][key].__class__.__name__}).{k}'] = p
    def set_layer(self, idx:int, layer:dict[str,Any]):
        if idx>=len(self.layers):
            raise IndexError
        self.layers[idx] = LayerWrapper(layer, idx)
        for key,module in layer.items():
            self.set_module(idx, key, module)
    
    def to(self, device=None, dtype=None):
        device, dtype = device_dtype_check(device, dtype)
        if (self.device==device or device is None) and (self.dtype == dtype or dtype is None):
            return self
        new = self.clone()
        for i in range(self.num_layers):
            for k in new.modules.keys():
                if not isinstance(new[i][k], To) and hasattr(new[i][k], 'to'):
                    new.set_module(i, k, new.layers[i][k].to(device=device or self.device, dtype=dtype or self.dtype))
        for k in new.modules.keys():
            if hasattr(new[k], 'to'):
                new.modules[k] = new.modules[k].to(device=device or self.device, dtype=dtype or self.dtype)
        return new
    
    def init_like(self):
        if self.is_dict:
            return Layers(self.modules, num_layers=self.num_layers, device=self.device, dtype=self.dtype)
        return Layers(*list(self.modules.values()), num_layers=self.num_layers, device=self.device, dtype=self.dtype)
    
    def clone(self):
        new = self.init_like()
        for i in range(self.num_layers):
            for k,m in self.layers[i].items():
                if hasattr(m, 'clone'):
                    new[i][k] = m.clone()
                else:
                    new[i][k] = m
        return new
    
    def state_dict(self):
        return {i:{k:v.state_dict() for k,v in layer.items() if hasattr(v, 'state_dict')} for i,layer in enumerate(self.layers)}
    
    def load_state(self, state):
        for i,layer in state.items():
            for k,m in layer.items():
                if not hasattr(self[i][k], 'load_state'):
                    raise ValueError(
                            '<<Layers.load_state>>\n'
                        f"Layer {i}, Module {k} has not load_state method.\n"
                        "Cannot load state for this layer.\n"
                        f'(layer keys: {list(layer.keys())})'
                    )
                self[i][k].load_state(m)
                
    def __repr__(self):
        return (
                    'Layers(\n'
            f'        device={self.device},\n'
            f'        dtype={self.dtype},\n'
            f'        num_layers={self.num_layers},\n'
            '        <<<modules>>>\n        '
            +'\n        '.join([f'[{i}]: {module}' for i,module in self.modules.items()])
            +'\n    )'
        )