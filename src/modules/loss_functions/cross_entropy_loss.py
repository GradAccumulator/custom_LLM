import torch
from src.utils.types import PathLike, Path, ListLike, Tensor
from .loss_function import LossFunction
from src.utils.torch_utils import device_dtype_check

class CrossEntropyLoss(LossFunction):
    def __init__(self,*, 
            logit_input             =True, 
            ignore_label_indices    =[], 
            dim                     =-1, 
            weight_device                  ='cpu',
            stability_scale         =1.0,
            weight:Path|PathLike|ListLike|dict=None, 
        ):
        self.logit_input = logit_input
        self.ignore_indices = torch.tensor(ignore_label_indices, device=weight_device)
        self.dim = dim
        if logit_input:
            self.forward = self._CE_softmax
        else:
            self.forward = self._CE       
        self.weight_device = weight
        self.stability_scale = stability_scale
        self.recip_stability_scale = 1.0/self.stability_scale
        self.device = weight_device
        
    def to(self, device=None, dtype=None):
        _,dtype = device_dtype_check(device, dtype)
        if self.dtype==dtype or dtype is None:
            return self
        new= CrossEntropyLoss(logit_input=self.logit_input, ignore_label_indices=self.ignore_indices, dim=self.dim, weight_device=self.device, stability_scale=self.stability_scale, weight=self.weight_device)
        return new
        
    def __repr__(self):
        return (
            f'CrossEntropyLoss(\n\t'
            f'logit_input={self.logit_input}, \n\t'
            f'ignore_indices={self.ignore_indices.tolist()}, \n\t'
            f'dim={self.dim}, \n\t'
            f'weight={self.weight_device}, \n\t'
            f'device={self.device}, \n\t'
            f'stability_scale={self.stability_scale}\n'
            ')'
        )
        
    def _gather(self, logits:Tensor, label:Tensor):
        gathered = logits.gather(self.dim, label)
        return gathered
    
    def _logsumexp(self, scaled:Tensor):
        exps = scaled.exp()
        sum = exps.sum(dim=self.dim, keepdim=True)
        logs = sum.log()
        return logs
    
    def _apply_weight(self, loss:Tensor, label:Tensor):
        if self.weight_device is not None:
            loss = loss*self.weight_device[label].to(loss.dtype)
        return loss
    
    def _apply_scales(self, loss:Tensor):
        if self.stability_scale != 1.0:
            loss = loss*self.stability_scale
        return loss
    
    def _apply_ignore_indices(self, loss:Tensor, label:Tensor):
        if self.ignore_indices != []:
            label_not_ignore = ~torch.isin(label, self.ignore_indices)
            loss = loss * label_not_ignore.to(loss.dtype)
            loss = loss.sum()/label_not_ignore.sum()
        else:
            loss = loss.mean()
        return loss
    
    def _apply_unreduction_scale(self, loss:Tensor):
        if self.stability_scale != 1.0:
            loss = loss*self.recip_stability_scale
        return loss
    
    def _CE_softmax(self, logits:Tensor, label:Tensor):
        label = label.unsqueeze(self.dim)
        scaled = logits - logits.max(dim=self.dim, keepdim=True).values
        
        gathered = self._gather(scaled, label)
        
        logs = self._logsumexp(scaled)
        
        loss = logs-gathered
        loss = self._apply_weight(loss, label)
        loss = self._apply_scales(loss)
        
        loss = self._apply_ignore_indices(loss, label)
        
        return self._apply_unreduction_scale(loss)
    
    def _CE(self, probs:Tensor, label:Tensor):
        label = label.unsqueeze(self.dim)
        gathered = self._gather(probs, label)
        
        loss = -gathered.log()
        
        loss = self._apply_weight(loss, label)
        loss = self._apply_scales(loss)
        loss = self._apply_ignore_indices(loss, label)
        
        return self._apply_unreduction_scale(loss)

    @property
    def device(self):
        return self._weight.device
    @property
    def weight_device(self):
        return self._weight.device
    @weight_device.setter
    def weight_device(self, loss_weights:Path|PathLike|ListLike|dict):
        if isinstance(loss_weights, PathLike) or isinstance(loss_weights, Path):
            self._weight = torch.load(loss_weights).half()
        elif type(loss_weights) == Tensor:
            self._weight = loss_weights.half()
        elif isinstance(loss_weights, ListLike):
            self._weight = Tensor(loss_weights).half()
        elif loss_weights is None:
            self._weight = None
        elif isinstance(loss_weights, dict):
            self._weight = torch.ones(loss_weights['num_classes'], device=self.device, dtype=torch.float16)
            for k,v in loss_weights.items():
                if k != "num_classes":
                    self._weight[int(k)] = v
        else:
            raise TypeError(f"[CE] loss_weights 타입({type(loss_weights)})은 유효하지 않습니다")

    def state_dict(self):
        weight = self._weight.cpu().detach().clone() if self.weight_device is not None else None
        return {
            'ignore_values': self.ignore_indices,
            'dim': self.dim,
            'loss_weights': weight,
            'device': self.device
        }
    
    @staticmethod
    def load_state(state:dict):
        return CrossEntropyLoss(
            logit_input=True, 
            ignore_label_indices=state['ignore_values'], 
            weight=state['loss_weights'],
            weight_device=state['device']
        )
    