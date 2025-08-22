


class LinearWarmup:
    def __init__(self, warmup_steps, max_lr, min_lr):
        self.warmup = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def step(self, current_step):
        return self.min_lr + (self.max_lr - self.min_lr) * current_step / self.warmup