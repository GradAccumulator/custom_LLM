import torch
import threading
from src.utils.processing.data_processing import shuffle_dataset

class DataLoader:
    def __init__(self, inputs:list[torch.Tensor], labels:list[torch.Tensor],/,*, device='cpu',save_type=torch.int16):
        self.inputs = inputs
        self.labels = labels
        
        self._length     = len(inputs)
        self._batch_size = len(inputs[0])
        self._size_last  = len(inputs[-1])
        self._total_size = (self._length-1)*self._batch_size+self._size_last

        self.device = device
        
        self._idx  = 0
    
    def __iter__(self):
        self._idx = 0
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._idx>=self._length:
            self._idx = 0
            raise StopIteration
        
        self._idx+=1
        
        return (
            self.inputs[self._idx-1].to(self.device, dtype=torch.int64), 
            self.labels[self._idx-1].to(self.device, dtype=torch.int64),
        )
    
    def shuffle(self):
        self.inputs,self.labels = shuffle_dataset(self.inputs, self.labels)
    
    def reset(self):
        self._idx = 0
    
    def __call__(self):
        return self.__next__()

    @property
    def length(self):
        return self._length
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def size_last(self):
        return self._size_last
    @property
    def total_size(self):
        return self._total_size
    @property
    def idx(self):
        return self._idx
    def is_last_batch(self):
        return self._idx==(self._length-1)
    def current_size(self):
        return len(self.inputs[self._idx-1])
    def current_size_rate(self):
        return self.current_size()/self._total_size
    def __len__(self):
        return self._length

    def state_dict(self):
        inputs = []
        labels = []
        for i in range(self._length):
            inputs.append(self.inputs[i].cpu().detach().clone())
            labels.append(self.labels[i].cpu().detach().clone())
        state = {
            'inputs':self.inputs,
            'labels':self.labels,
            'idx'   :self._idx,
            'device':self.device,
        }
        return state
    @staticmethod
    def load_state(state):
        loader = DataLoader(state["inputs"], state["labels"], device='cuda')
        loader._idx = state['idx']
        return loader
    