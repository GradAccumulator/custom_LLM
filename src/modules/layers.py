import torch
from typing import overload, Any
from src.utils.torch_utils import device_dtype_check
from src.modules.to import To


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
        
    @staticmethod
    def wrapping_layers(layers:list[dict[str,Any]]):
        for i in range(len(layers)):
            layers[i] = LayerWrapper(layers[i], i)
        return layers
    
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
        
        self.modules:dict[str, Any] = modules
        self.dtype = dtype
        self.device = device
        self.num_layers = num_layers
        self._set_layers(modules)

    def _set_layers(self, modules:dict):
        self.layers = [{} for _ in range(self.num_layers)]
        for i in range(self.num_layers):
            for k,m in modules.items():
                if hasattr(m, 'init_like'):
                    m = m.init_like()
                if not isinstance(m, To) and hasattr(m, 'to'):
                    self.layers[i][k] = m.to(device=self.device, dtype=self.dtype)
                else:
                    self.layers[i][k] = m
            self.layers[i] = LayerWrapper(self.layers[i], i)
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
        return self.params_as_dict
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
    def __setitem__(self, idx:int, module:dict[str,Any]):
        if idx>=len(self.layers):
            raise IndexError
        self.layers[idx] = module
    
    def to(self, device=None, dtype=None):
        device, dtype = device_dtype_check(device, dtype)
        if (self.device==device or device is None) and (self.dtype == dtype or dtype is None):
            return self
        new = self.clone()
        for i in range(self.num_layers):
            for k in new.modules.keys():
                if not isinstance(new[i][k], To) and hasattr(new[i][k], 'to'):
                    new[i][k] = new[i][k].to(device=device or self.device, dtype=dtype or self.dtype)
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