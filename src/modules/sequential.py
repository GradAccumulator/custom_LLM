import torch
from typing import overload, Any
from src.utils.torch_utils import 보이면_바로_교체하기
from src.modules.to import To
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
    def __init__(self, *layers, device='cpu', dtype=torch.float32):...
    @overload
    def __init__(self, layers:tuple|list, device='cpu', dtype=torch.float32):...
    def __init__(self, *layers_list, device='cpu', dtype=torch.float32,**layers_dict):
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
        
        super().__init__(device=device, dtype=dtype)
        self.layers = layers
        self._is_training = True
        self.inputs = {}

        for k in self.layers.keys():
            if not isinstance(self.layers[k], To) and hasattr(self.layers[k], 'to'):
                self[k] = self.layers[k].to(device=self._device, dtype=self._dtype)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        try:
            for key,layer in self.layers.items():
                #if not isinstance(layer, To): self.inputs[key] = x
                x = layer(x)
            return x
        except Exception as e:
            print(f"error in Sequential layer {key}")
            raise e
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device=None, dtype=None):
        device, dtype = 보이면_바로_교체하기(device, dtype)
        if (device is None and dtype is None) or (self._device == device and self._dtype == dtype):
            return self
        seq = self.clone()
        device = device or self._device
        dtype = dtype or self._dtype
        for k in seq.layers.keys():
            if not isinstance(seq[k], To) and hasattr(seq[k], 'to'):
                seq[k] = seq[k].to(device=device, dtype=dtype)
        seq._device = device
        seq._dtype = dtype
        return seq
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx:int) -> dict[str,Any]:
        if idx not in self.layers.keys():
            raise IndexError(idx)
        return self.layers[idx]
    def __setitem__(self, idx:int|str, value:dict[str,Any]):
        if idx not in self.layers.keys():
            raise KeyError(idx)
        if isinstance(value, BaseModule):
            pop_list = []
            for k in self._buffer.keys():
                if k[k.find('[')+1:k.find(']')] == str(idx):
                    pop_list.append(k)
            for k in pop_list:
                self._buffer.pop(k)
            for k,v in value._buffer.items():
                self._buffer[f'Sequantial[{idx}].{k}'] = v
        self.layers[idx] = value
    def keys(self):
        return self.layers.keys()
    def values(self):
        return self.layers.values()
    def items(self):
        return self.layers.items()
    
    def _shallow_copy(self):
        if self.is_dict:
            seq = Sequential(self.layers, device=self._device, dtype=self._dtype)
        else:
            seq = Sequential(*list(self.layers.values()), device=self._device, dtype=self._dtype)
        return seq
        
    
    def clone(self):
        seq = self._shallow_copy()
        for k in self.layers.keys():
            if hasattr(seq[k], 'clone'):
                seq[k] = seq[k].clone()
        return seq
    
    def init_like(self):
        seq = self._shallow_copy()
        for k in self.layers.keys():
            if hasattr(seq[k], 'init_like'):
                seq[k] = seq[k].init_like()
        return seq
    
    def state_dict(self):
        return {k:layer.state_dict() for k,layer in self.layers.items() if hasattr(layer, 'state_dict')}
    
    def load_state(self, state):
        for k,s in state.items():
            if not hasattr(self[k], 'load_state'):
                raise ValueError(
                        '<<Sequential.load_state>>\n'
                    f"Layer {k} has not load_state method.\n"
                    f"Cannot load state for this layer.\n"
                    f'(state keys: {list(state.keys())})'
                )
            self[k].load_state(s)
            
    def eval(self):
        self._is_training = False
        for layer in self.layers.values():
            if hasattr(layer, 'eval'):
                layer.eval()
                
    def train(self):
        self._is_training = True
        for layer in self.layers.values():
            if hasattr(layer, 'train'):
                layer.train()
                
    @property
    def mode(self):
        return 'train' if self._is_training else 'eval'
    @property
    def is_training(self):
        return self._is_training
    
    def __repr__(self):
        return super().__repr__((
            f'Sequential(\n'
            f'    device={self._device},\n'
            f'    dtype={self._dtype},\n'
            f'    <<layers>>\n    '
            +'\n    '.join([f'{i}: {module}' for i,module in self.layers.items()])
            +'\n)'
        ))