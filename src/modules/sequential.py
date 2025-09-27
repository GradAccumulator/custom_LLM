import torch
from typing import overload, Any
from .base_module import BaseModule

class Sequential(BaseModule):
    '''
    sequential에 넣을 클래스에 있어야하는거 \n
    to() : device, dtype 옵션 있으면, 그리고 device dtype 둘 다 있어야함. 안써도 받아야함\n
    clone,init_like : 파라미터 있으면\n
    load_state : 파라미터 있으면\n
    state_dict : 파라미터 있으면\n
    '''
    @overload
    def __init__(self,*, device='cpu', dtype=torch.float32,**layers):...
    @overload
    def __init__(self, *layers, device='cpu', dtype=torch.float32,**_):...
    @overload
    def __init__(self, layers:tuple|list, device='cpu', dtype=torch.float32):...
    def __init__(self, *layers_list, device='cpu', dtype=torch.float32,**layers_dict):
        super().__init__(device, dtype, *layers_list, **layers_dict, device=device, dtype=dtype)
        if layers_dict and layers_list:
            raise ValueError('layers_dict와 layers_list를 동시에 전달할 수 없습니다.')
        elif not layers_dict and not layers_list:
            raise ValueError('layers_dict and layers_list are empty.')
        elif len(layers_list) >= 1 and isinstance(layers_list[0], dict):
            if len(layers_list) > 1:
                raise ValueError('layers를 딕셔너리로 전달할 때는 여러개의 인자를 전달할 수 없습니다.')
            self.is_dict = True
            layers = layers_list[0]
        elif layers_dict:
            layers = layers_dict
            self.is_dict = True
        else:
            layers = dict(enumerate(layers_list))
            self.is_dict = False
        
        self.layers  :dict[str, BaseModule]|dict[int, BaseModule] = layers

        self.to(device=self.device, dtype=self.dtype)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        try:
            for key,layer in self.layers.items():
                x = layer(x)
            return x
        except Exception as e:
            print(f"error in Sequential layer {key}")
            raise e
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx:int|str) -> dict[str,Any]:
        if idx not in self.layers.keys():
            raise IndexError(idx)
        return self.layers[idx]
    def keys(self):
        return self.layers.keys()
    def values(self):
        return self.layers.values()
    def items(self):
        return self.layers.items()
    
    def __repr__(self):
        return (
            f'Sequential(\n'
            f'    device={self._device},\n'
            f'    dtype={self._dtype},\n'
            f'    <<layers>>\n    '
            +'\n    '.join([f'{i}: {module}' for i,module in self.layers.items()])
            +'\n)'
        )