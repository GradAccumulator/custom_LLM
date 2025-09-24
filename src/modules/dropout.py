import torch
import src.c_modules as _C

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self._is_training = True
        self.inv_p = 1/(1-p)
        
    def forward(self, x):
        return _C.dropout(
            x,
            self.p*self.is_training
        )
        
    @property
    def mode(self):
        return 'train' if self.is_training else 'eval'
    @property
    def is_training(self):
        return self._is_training
    def eval(self):
        self._is_training = False
    def train(self):
        self._is_training = True
    @property
    def p(self):
        return self._p
    @p.setter
    def p(self, value):
        self._p = value
        self.inv_p = 1/(1-value)
        
    def init_like(self):
        return Dropout(p=self.p)
    def clone(self):
        new = Dropout(p=self.p)
        new._is_training = self._is_training
        return new
    
    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"Dropout(p={self.p}, mode={self.mode})"