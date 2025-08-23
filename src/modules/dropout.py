import torch

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self._is_training = True
        
    def forward(self, x):
        if not self._is_training:
            return x
        mask = (torch.rand_like(x, dtype=torch.float16) < self.p)
        return x * mask
        
    @property
    def mode(self):
        return self._is_training
    def eval(self):
        self._is_training = False
    def train(self):
        self._is_training = True
        
    def init_like(self):
        return Dropout(p=self.p)
    def clone(self):
        new = Dropout(p=self.p)
        new._is_training = self._is_training
        return new
    
    def __call__(self, x):
        return self.forward(x)