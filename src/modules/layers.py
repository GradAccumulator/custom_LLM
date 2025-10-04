import torch
from typing import overload, Any
#from src.utils.torch_utils import _
from src.modules.to import To
from .base_module import Module


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
    
    def __getitem__(self, key:int|str):
        return super().__getitem__(str(key))
    def __setitem__(self, key:int|str, module):
        super().__setitem__(str(key), module)
    
    @staticmethod
    def wrap_layers(layers:list[dict[str,Any]]):
        for i in range(len(layers)):
            layers[i] = LayerWrapper(layers[i], i)
        return layers

class Layers(Module):
    @overload
    def __init__(self, modules:dict[str, Any], num_layers, device='cpu', dtype=torch.float32, save_inputs=False):...
    @overload
    def __init__(self, *modules, num_layers, device='cpu', dtype=torch.float32, save_inputs=False):...
    @overload
    def __init__(self, *, num_layers, device='cpu', dtype=torch.float32, save_inputs=False, **modules):...
    def __init__(self, *modules_list, num_layers, device='cpu', dtype=torch.float32, save_inputs=False, **modules_dict):
        super().__init__(device, dtype, *modules_list, num_layers, device, dtype, save_inputs, **modules_dict)
        
        if modules_dict and modules_list:
            raise ValueError('modules_dict와 modules_list를 동시에 전달할 수 없습니다.')
        elif not modules_dict and not modules_list:
            raise ValueError('modules_dict와 modules_list가 모두 비어있습니다.')
        elif len(modules_list) >= 1 and isinstance(modules_list[0], dict):
            if len(modules_list) > 1:
                raise ValueError('modules를 딕셔너리로 전달할 때는 여러개의 인자를 전달할 수 없습니다.')
            self.is_dict = True
            modules = modules_list[0]
        elif modules_dict:
            modules = modules_dict
            self.is_dict = True
        else:
            modules = {}
            for i,m in enumerate(modules_list):
                modules[str(i)] = m
            self.is_dict = False
        #modules 인자가 list이든 dict이든 상관없이 
        #self.modules:dict[str, BaseModule]로 저장함
        
        super().setattr('modules', modules)
        self.modules:dict[str, Module]
        
        self.num_layers = num_layers
        self.layers = [{} for _ in range(self.num_layers)]
        self._set_layers(modules)
        self.save_inputs = save_inputs
        if self.save_inputs:
            self.inputs = {}
    
    def forward(self, x):
        try:
            for layer in self.layers:
                for key,module in layer.items():
                    #self.inputs[key] = x
                    x = module(x)
            return x
        except Exception as e:
            print(f"error in layer {layer.idx}, module {key}")
            raise e
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx:int|str) -> dict[str,Any]:
        if idx>=len(self.layers):
            raise IndexError
        return self.layers[idx]
    
    def _set_layers(self, modules:dict[str,Module]):
        for i in range(self.num_layers):
            for k,m in modules.items():
                if isinstance(m, Module):
                    m = m.init_like()
                    for name,param in m._buffer.items():
                        self._buffer[f'layers[{i}][{k}]({m.__class__.__name__}).{name}'] = param
                    self._modules[f'layers[{i}][{k}]'] = m
                self.layers[i][k] = m
            self.layers[i] = LayerWrapper(self.layers[i], i)
    
    def __repr__(self):
        return (
                    'Layers(\n'
            f'        device={self._device},\n'
            f'        dtype={self._dtype},\n'
            f'        num_layers={self.num_layers},\n'
            f'        save_inputs={self.save_inputs},\n'
            '        <<<modules>>>\n        '
            +'\n        '.join([f'[{i}]: {module}' for i,module in self.modules.items()])
            +'\n    )'
        )