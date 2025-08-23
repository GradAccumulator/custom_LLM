from enum import Enum
class Phase(Enum):
    WARMUP = 'warmup'
    DECAY = 'decay'

class Scheduler:
    def __init__(self, optimizer, Warmup_type, warmup_steps, Decay_type, total_steps,*, min_lr=0.0):
        '''
        you can set decay type as None if you don't want to use
        '''
        self._max_lr = optimizer.lr
        self.warmup = Warmup_type(warmup_steps, self._max_lr, min_lr)
        self.decay = Decay_type(warmup_steps,total_steps, self._max_lr, min_lr)
        
        self._step = 0
        self.optimizer = optimizer
        self.optimizer.lr = min_lr
        self.prev_lr = min_lr
    
    def step(self):
        self._step += 1
        self.prev_lr = self.optimizer.lr
        if self._step < self.warmup_steps:
            self.optimizer.lr = self.warmup.step(self._step)
        elif self.decay:
            self.optimizer.lr = self.decay.step(self._step)
    
    @property
    def min_lr(self):
        return self.warmup.min_lr
    @min_lr.setter
    def min_lr(self, value):
        self.warmup.min_lr = value
        if self.decay:
            self.decay.min_lr = value
    
    @property
    def warmup_steps(self):
        return self.warmup.warmup
    @warmup_steps.setter
    def warmup_steps(self, value):
        self.warmup.warmup = value
        if self.decay:
            self.decay.warmup = value
            
    @property
    def max_lr(self):
        return self._max_lr
    @max_lr.setter
    def max_lr(self, value):
        self._max_lr = value
        self.warmup.max_lr = value
        if self.decay:
            self.decay.max_lr = value
    
    @property
    def total_steps(self):
        if self.decay:
            return self.decay.total
        else:
            return None
    @total_steps.setter
    def total_steps(self, value):
        if self.decay:
            self.decay.total = value
        else:
            raise AttributeError("total_steps can only be set if decay type is not None")
    
    @property
    def decay_type_name(self):
        if self.decay:
            return self.decay.__class__.__name__
        else:
            return None
    @property
    def warmup_type_name(self):
        return self.warmup.__class__.__name__
    @property
    def warmup_type(self):
        return type(self.warmup)
    @property
    def decay_type(self):
        return type(self.decay) if self.decay else None

    @property
    def lr(self):
        return self.optimizer.lr
    @lr.setter
    def lr(self, value):
        self.optimizer.lr = value
        
    @property
    def current_step(self):
        return self._step
    
    @property
    def current_phase(self):
        if self._step < self.warmup_steps:
            return Phase.WARMUP
        elif self.decay:
            return Phase.DECAY
        else:
            return None
    
    def state_dict(self):
        return {
            'max_lr':self.max_lr,
            'min_lr':self.min_lr,
            'step':self._step,
            'warmup_steps':self.warmup_steps,
            'total_steps':self.total_steps,
            'current_lr':self.lr
        }
        
    @staticmethod
    def load_state(optimizer, warmup_type, decay_type, state):
        scheduler = Scheduler(optimizer, warmup_type, state['warmup_steps'], decay_type, state['total_steps'], min_lr=state['min_lr'])
        scheduler._step = state['step']
        scheduler.max_lr = state['max_lr']
        optimizer.lr = state['current_lr']
        return scheduler
    
    def __repr__(self):
        return (
            f'Scheduler(\n'
            f'    warmup_type={self.warmup},\n'
            f'    warmup_steps={self.warmup_steps},\n'
            f'    decay_type={self.decay},\n'
            f'    total_steps={self.total_steps},\n'
            f'    min_lr={self.min_lr},\n'
            f'    max_lr={self.max_lr},\n'
            f'    current_lr={self.lr},\n'
            f'    current_step={self._step},\n'
            f'    current_phase={self.current_phase}\n'
            f')'
        )