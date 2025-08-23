import math

class CosineDecay:
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr):
        self.warmup = warmup_steps
        self.total = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def step(self, current_step):
        return self.min_lr + 0.5*(self.max_lr-self.min_lr)*(1+math.cos((current_step/self.total)*math.pi))
    
    def __repr__(self):
        return 'CosineDecay'