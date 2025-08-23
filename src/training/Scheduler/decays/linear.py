


class LinearDecay:
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr):
        self.warmup = warmup_steps
        self.total = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, current_step):
        return max(self.min_lr, ((self.total - current_step) / (self.total-self.warmup)) * self.max_lr)
    
    def __repr__(self):
        return 'LinearDecay'