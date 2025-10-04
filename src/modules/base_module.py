from __future__ import annotations
import torch
import math
from typing import (
    overload, 
    Optional, 
    Callable, 
    Any, 
    NewType, 
    Literal, 
    TypeVar, 
    Self
)
from src.utils.dev_utils import run_on_exit_if_true, is_valid_varname
from src.utils.torch_utils import restore_param, device_dtype_check

def isparameter(tensor, error_msg:Optional[str]=None)->bool:
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
        if not tensor.is_leaf:
            raise ValueError(error_msg or f"tensor {tensor} is set to requires grad but not leaf.")
        else:
            return True
    return False

def is_parameter_or_module(tensor):
    return isparameter(tensor) or isinstance(tensor, Module)

MODULE_T = TypeVar('MODULE_T', bound='Module')
class Module(object):
    def __init__(self, device_='cpu', dtype_=torch.float32,/, *init_args, **init_kwargs) -> None:
        super().__setattr__("_device"          , torch.device(device_) if device_ else device_)
        super().__setattr__("_dtype"           , dtype_)
        super().__setattr__("_buffer"          , {})
        super().__setattr__("_modules"         , {})
        super().__setattr__("_parameter_dicts" , {})
        super().__setattr__("_parameter_lists" , {})
        super().__setattr__("_parameters"      , {})
        super().__setattr__("_others"          , {})
        super().__setattr__("_added_to__apply" , {})
        self._device            :torch.device
        self._dtype             :torch.dtype
        self._buffer            :dict[str, torch.Tensor]
        self._modules           :dict[str, Module]
        self._parameter_dicts   :dict[str, dict[str,torch.Tensor|Module]]
        self._parameter_lists   :dict[str, list[torch.Tensor|Module]]
        self._parameters        :dict[str, torch.Tensor]
        self._others            :dict[str, Any]
        self._added_to__apply   :dict[str, torch.Tensor]
        
        super().__setattr__("_init_args"  , init_args)
        super().__setattr__("_init_kwargs", init_kwargs)
        self._init_args      :tuple[Any]
        self._init_kwargs    :dict[str, Any]
        
        super().__setattr__("_is_training"   , True)
        super().__setattr__("_is_inferencing", False)
        self._is_training    :bool
        self._is_inferencing :bool
        
    def add_to_state(self, name:str, value:Any)->None:
        self._others[name] = value
    def add_to__apply(self, name:str, value:torch.Tensor)->None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"[{self.__class__.__name__}.add_to__apply()] Expected torch.Tensor but got {type(value)}")
        self._added_to__apply[name] = value
    
    @overload
    def add_buffer(self, name:str, param:torch.Tensor)->None:...
    @overload
    def add_buffer(self, name:str, module:Module)->None:...
    def add_buffer(self, name:str, param_or_module:torch.Tensor|Module)->None:
        if isinstance(param_or_module, Module):
            self._modules[name] = param_or_module
            for k,v in param_or_module._buffer.items():
                self.add_buffer(f'{name}.{k}', v)
        elif isparameter(param_or_module):
            self._buffer[name] = param_or_module
    
    def setattr(self, name:str, value:Any)->None:
        super().__setattr__(name, value)
    
    def __setattr__(self, name:str, value)->None:
        '''
        
        '''
        if not hasattr(self, '_buffer'):
            raise AttributeError("Must have to call super().__init__ before set parameters or modules")
        
        if isparameter(value):
            self.add_buffer(name, value)
            self._parameters[name] = value
        
        elif isinstance(value, Module):
            self.add_buffer(name, value)
        
        elif isinstance(value, dict):
            with run_on_exit_if_true(lambda:self._parameter_dicts.update({name:value})) as trigger:
                for k,v in value.items():
                    if is_parameter_or_module(v):
                        self.add_buffer(f'{name}[{k}]', v)
                        trigger.true()
        
        elif isinstance(value, list|tuple):
            with run_on_exit_if_true(lambda:self._parameter_lists.update({name:value})) as trigger:
                for i,v in enumerate(value):
                    if is_parameter_or_module(v):
                        self.add_buffer(f'{name}[{i}]', v)
                        trigger.true()
        
        super().__setattr__(name, value)

    def __delattr__(self, name:str)->None:
        '''
        
        '''
        if name in self._modules:
            self._modules.pop(name)
            for k in self._buffer.keys():
                if k.startswith(f'{name}.'):
                    self._buffer.pop(k)
        elif name in self._parameters:
            self._parameters.pop(name)
            self._buffer.pop(name)
        elif name in self._parameter_dicts:
            self._parameter_dicts.pop(name)
            for k in self._buffer.keys():
                if k.startswith(f"{name}["):
                    self._buffer.pop(k)
        elif name in self._parameter_lists:
            self._parameter_lists.pop(name)
            for k in self._buffer.keys():
                if k.startswith(f"{name}["):
                    self._buffer.pop(k)
        super().__delattr__(name)
    
    @torch.no_grad()
    def _init_parameter_he(self, parameter:torch.Tensor)->None:
        d_out = parameter.size(-2)
        parameter.mul_(math.sqrt(1/d_out))
    @torch.no_grad()
    def _init_parameter_xavier(self, parameter:torch.Tensor)->None:
        d_in = parameter.size(-1)
        d_out = parameter.size(-2)
        parameter.mul_(math.sqrt(2/(d_in+d_out)))
    
    @torch.no_grad()
    def _init_weights_fn(self, fn:Callable[[parameter_name, parameter], None]) -> None:
        with torch.no_grad():
            for parameter_name,parameter  in self._buffer.items():
                fn(parameter_name, parameter)

    parameter_name = NewType("parameter_name", str)
    parameter = NewType("parameter", torch.Tensor)
    @overload
    def init_weights(self, init_method=None)->None:...
    @overload
    def init_weights(self, init_method:str='none')->None:...
    @overload
    def init_weights(self, init_method_fn:Callable[[parameter_name, parameter],Any])->None:...
    @torch.no_grad()
    def init_weights(self, init_method:Optional[str|Callable[[parameter_name, parameter], None]]='none') -> None:
        '''

        '''
        if isinstance(init_method, Callable):
            self._init_weights_fn(init_method)
            return
        
        init_method = init_method or 'none'
        init_method = init_method.lower()
        
        if init_method=='he':
            self._init_weights_fn(lambda parameter_name, parameter:self._init_parameter_he(parameter) if 'weight' in parameter_name else None)
        elif init_method=='xavier':
            self._init_weights_fn(lambda parameter_name, parameter:self._init_parameter_xavier(parameter) if 'weight' in parameter_name else None)
    
    @overload
    def init_parameter(self, parameter:torch.Tensor, init_method=None)->None:...
    @overload
    def init_parameter(self, parameter:torch.Tensor, init_method:str='none')->None:...
    @overload
    def init_parameter(self, parameter:torch.Tensor, init_value:torch.Tensor)->None:...
    @torch.no_grad()
    def init_parameter(self, parameter:torch.Tensor, init_method:Optional[str|torch.Tensor]='none') -> None:
        '''

        '''
        if isinstance(init_method, torch.Tensor):
            parameter.data = init_method
            return
        
        init_method = (str(init_method)).lower()
        
        if init_method=='he':
            self._init_parameter_he(parameter)
        elif init_method=='xavier':
            self._init_parameter_xavier(parameter)
    
    new_parameter = NewType("new_parameter", torch.Tensor)
    @torch.no_grad()
    def _apply(self, fn:Callable[[parameter], new_parameter])->Self:
        for module in self._modules.values():
            module._apply(fn)
        for dict in self._parameter_dicts.values():
            for param in dict.values():
                if isinstance(param, torch.Tensor):
                    param.data = (fn(param))
                elif isinstance(param, Module):
                    param._apply(fn)
        for param_list in self._parameter_lists.values():
            for param in param_list:
                if isinstance(param, torch.Tensor):
                    param.data = (fn(param))
                elif isinstance(param, Module):
                    param._apply(fn)
        for param in self._parameters.values():
            param.data = (fn(param))
        for value in self._others.values():
            if isinstance(value, torch.Tensor):
                value.data = (fn(value))
        for value in self._added_to__apply.values():
            value.data = (fn(value))
        return self
    
    def __call__(self, *args, **kwargs):
        if not hasattr(self,'forward'):
            raise AttributeError(f"[{self.__class__.__name__}] forward() method is required.")
        return self.forward(*args, **kwargs)
    
    def to(self, *args, **kwargs) -> Self:
        self._device,self._dtype = device_dtype_check(*args, **kwargs, default_device=self._device, default_dtype=self._dtype)
        for module in self._modules.values():
            module._device,module._dtype = self._device,self._dtype
        return self._apply(lambda t: t.to(*args, **kwargs))
    def cuda(self) -> Self:
        self._device = torch.device('cuda')
        for module in self._modules.values():
            module._device = torch.device('cuda')
        return self._apply(lambda t: t.cuda())
    def cpu(self) -> Self:
        self._device = torch.device('cpu')
        for module in self._modules.values():
            module._device = torch.device('cpu')
        return self._apply(lambda t: t.cpu())
    def half(self) -> Self:
        self._dtype = torch.half
        for module in self._modules.values():
            module._dtype = torch.half
        return self._apply(lambda t: t.half())
    def float(self) -> Self:
        self._dtype = torch.float
        for module in self._modules.values():
            module._dtype = torch.float
        return self._apply(lambda t: t.float())
    def train(self) -> Self:
        self._is_training = True
        self._is_inferencing = False
        for module in self._modules.values():
            module.train()
        return self
    def eval(self) -> Self:
        self._is_training = False
        self._is_inferencing = False
        for module in self._modules.values():
            module.eval()
        return self
    def inference(self) -> Self:
        self._is_training = False
        self._is_inferencing = True
        for module in self._modules.values():
            module.inference()
        return self
    
    @property
    def is_training(self) -> bool:
        return self._is_training
    @property
    def is_inferencing(self) -> bool:
        return self._is_inferencing
    @property
    def mode(self) -> Literal['train', 'inference', 'eval']:
        if self._is_training:
            return 'train'
        elif self._is_inferencing:
            return 'inference'
        else:
            return 'eval'
    @property  
    def device(self) -> torch.device:
        return self._device
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    def parameters(self) -> list[torch.Tensor]:
        return list(self._buffer.values())
    def named_parameters(self) -> dict[str, torch.Tensor]:
        return self._buffer
    def modules(self) -> list[Module]:
        return list(self._modules.values())
    def named_modules(self) -> dict[str, Module]:
        return self._modules
    
    def init_like(self)->Self:
        try:
            return self.__class__(*self._init_args, **self._init_kwargs)
        except (TypeError,ValueError) as e:
            import inspect
            raise e.__class__(
                f"[BaseModule of {self.__class__.__name__}] init_like() was called, but no or invalid initialization arguments were provided.\n"
                f"(reveal args:{inspect.signature(self.__class__.__init__)})\n"
                f"(saved init_args:{self._init_args}, saved init_kwargs:{self._init_kwargs})"
            ) #from e
    
    DictParams =NewType('DictParams' , dict[str, dict[str, torch.Tensor]])
    ListParams =NewType('ListParams' , dict[str, dict[int, torch.Tensor]])
    Modules    =NewType('Modules'    , dict[str, "Module"])
    Params     =NewType('Params'     , dict[str, torch.Tensor])
    Others     =NewType('Others'     , dict[str, Any])
    StateDict  =dict[str, DictParams|ListParams|Modules|Params|Others]
    def state_dict(self)->StateDict:
        StateDict = self.StateDict
        state:StateDict = {
            'dicts':{},
            'lists':{},
            'modules':{},
            'parameters':{},
            'others':self._others
        }
        for name,module in self._modules.items():
            if is_valid_varname(name):
                state['modules'][name] = module.state_dict()
        for name,dict in self._parameter_dicts.items():
            for key,value in dict.items():
                if isparameter(value):
                    state['dicts'].setdefault(name,{}).setdefault(key,{})['data'] = value.cpu()
                    state['dicts'][name][key]['grad'] = value.grad.cpu() \
                        if value.grad is not None else None
                elif isinstance(value, Module):
                    state['dicts'].setdefault(name,{})[key] = value.state_dict()
        for name,list in self._parameter_lists.items():
            for i,value in enumerate(list):
                if isparameter(value):
                    state['lists'].setdefault(name,{}).setdefault(i,{})['data'] = value.cpu()
                    state['lists'][name][i]['grad'] = value.grad.cpu() \
                        if value.grad is not None else None
                if isinstance(value, Module):
                    state['lists'].setdefault(name,{})[i] = value.state_dict()
        for name,param in self._parameters.items():
            state['parameters'].setdefault(name,{})['data'] = param.cpu()
            state['parameters'][name]['grad'] = param.grad.cpu() \
                if param.grad is not None else None
        return state
    
    @torch.no_grad()
    def load_state(self, state:StateDict)->None:
        prefix = f'[{self.__class__.__name__}.load_state()] Invalid key'
        def MakeKeyError(position, *keys):
            return KeyError(f"{prefix} {keys} in self.{position}")
        
        try: 
            for key,module_state in state['modules'].items():
                self._modules[key].load_state(module_state)
        except KeyError as e:
            raise MakeKeyError('_modules', key) #from e
        
        try:
            for key,dict_states in state['dicts'].items():
                for k,dict_state in dict_states.items():
                    if isparameter(self._parameter_dicts[key][k]):
                        self._parameter_dicts[key][k].data = (restore_param(dict_state['data'], self.device, self.dtype))
                        self._parameter_dicts[key][k].grad = dict_state['grad'].to(self.device, self.dtype) \
                            if dict_state['grad'] is not None else None
                    elif isinstance(self._parameter_dicts[key][k], Module):
                        self._parameter_dicts[key][k].load_state(dict_state)
        except KeyError as e:
            raise MakeKeyError('_parameter_dicts', key,k) #from e
        
        try:
            for key,list_states in state['lists'].items(): 
                for i,list_state in list_states.items():
                    if isparameter(self._parameter_lists[key][i]):
                        self._parameter_lists[key][i].data = (restore_param(list_state['data'], self.device, self.dtype))
                        self._parameter_lists[key][i].grad = list_state['grad'].to(self.device, self.dtype) \
                            if list_state['grad'] is not None else None
                    elif isinstance(self._parameter_lists[key][i], Module):
                        self._parameter_lists[key][i].load_state(list_state)
        except KeyError as e:
            raise MakeKeyError('_parameter_lists', key,i) #from e
        
        try:
            for key,param_state in state['parameters'].items():
                self._parameters[key].data = (restore_param(param_state['data'], self.device, self.dtype))
                self._parameters[key].grad = param_state['grad'].to(self.device, self.dtype) \
                    if param_state['grad'] is not None else None
        except KeyError as e:
            raise MakeKeyError('_parameters', key) #from e
        
        for key,other_state in state['others'].items():
            if not key in self.__dict__:
                raise MakeKeyError('others', key)
            self.__dict__[key] = other_state
        