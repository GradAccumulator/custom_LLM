import torch
from typing import overload, Any
from src.utils.torch_utils import device_dtype_check

class Sequential:
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
    def __init__(self, layers:dict, device='cpu', dtype=torch.float32):...
    def __init__(self, *layers_list:tuple, device='cpu', dtype=torch.float32,**layers_dict:dict):
        if layers_dict and layers_list:
            raise ValueError('layers_dict와 layers_list를 동시에 전달할 수 없습니다.')
        elif not layers_dict and not layers_list:
            raise ValueError('layers_dict와 layers_list가 모두 비어있습니다.')
        elif isinstance(layers_list[0], dict):
            self.is_dict = True
            layers = layers_list[0]
            if len(layers_list) > 1:
                raise ValueError('layers를 딕셔너리로 전달할 때는 여러개의 인자를 전달할 수 없습니다.')
        elif layers_dict:
            layers = layers_dict
            self.is_dict = True
        else:
            layers = dict(enumerate(layers_list))
            self.is_dict = False
        
        self.layers = layers
        self.device = device
        self.dtype = dtype
        self._is_training = True

        for k in self.layers.keys():
            if hasattr(self.layers[k], 'to'):
                self.layers[k] = self.layers[k].to(device=self.device, dtype=self.dtype)
                
        self.params_as_dict = {k:layer.parameters() for k,layer in self.layers.items() if hasattr(layer, 'parameters')}
        self.params_as_list = [p for layer in self.layers.values() if hasattr(layer, 'parameters') for p in layer.parameters()]
    
    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters_as_dict(self):
        return self.params_as_dict
    def parameters(self):
        return self.params_as_list
    
    def to(self, device=None, dtype=None):
        device, dtype = device_dtype_check(device, dtype)
        if (device is None and dtype is None) or (self.device == device and self.dtype == dtype):
            return self
        seq = self.clone()
        seq.device = device if device is not None else self.device
        seq.dtype = dtype if dtype is not None else self.dtype
        for k in seq.layers.keys():
            if hasattr(self.layers[k], 'to'):
                seq.layers[k] = seq.layers[k].to(device=device, dtype=dtype)
        return seq
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx:int) -> dict[str,Any]:
        if idx not in self.layers.keys():
            raise IndexError(idx)
        return self.layers[idx]
    def __setitem__(self, idx:int, value:dict[str,Any]):
        self.layers[idx] = value
    
    def _shallow_copy(self):
        if self.is_dict:
            seq = Sequential(self.layers, device=self.device, dtype=self.dtype)
        else:
            seq = Sequential(*list(self.layers.values()), device=self.device, dtype=self.dtype)
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
                    "Cannot load state for this layer.\n"
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
        return (
            f'Sequential(\n'
            f'    device={self.device},\n'
            f'    dtype={self.dtype},\n'
            f'    layers={self.layers}\n'
            ') : \n<<layers>>\n'
            +'\n    '.join([f'{i}: {module}' for i,module in self.layers.items()])
        )