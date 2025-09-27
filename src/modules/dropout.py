import torch
import src.c_modules as _C
from src.modules.base_module import BaseModule

class Dropout(BaseModule):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self._is_training = True
        
    def forward(self, x):
        return _C.dropout(
            x,
            self.p*self.is_training
        )
    
    def __repr__(self):
        return f"Dropout(p={self.p}, mode={self.mode})"