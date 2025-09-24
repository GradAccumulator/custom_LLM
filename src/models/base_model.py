import torch
from src.modules.base_module import BaseModule
from src.utils.torch_utils import device_dtype_check

class BaseModel:
    def __init__(self, device='cpu', dtype=torch.float32):
        self._parameters = {}
        self.device,self.dtype = device_dtype_check(device, dtype)
    
    def __setattr__(self, name, value):
        if not hasattr(self, '_parameters'):
            self._parameters = {}
        if isinstance(value, torch.Tensor) and value.requires_grad==True:
            if not value.is_leaf:
                raise ValueError(
                    f"value {name} in class {self.__class__.__name__} is set to requires grad but not leaf."
                )
            self._parameters[f"{self.__class__.__name__}.{name}"] = value
        elif isinstance(value, BaseModule):
            for n,p in value._parameters:
                self._parameters[f"{self.__class__.__name__}.{name}.{n}"] = p