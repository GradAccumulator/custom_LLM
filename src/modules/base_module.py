from __future__ import annotations
import torch
import math
from typing import overload, Optional, Callable, Any, NewType
from src.utils.dev_utils import run_on_exit_if_true, is_valid_varname
from src.utils.torch_utils import restore_param

def is_parameter(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.requires_grad

def is_parameter_or_module(tensor):
    return is_parameter(tensor) or isinstance(tensor, BaseModule)

class BaseModule(object):
    '''
    모듈에 있어야 하는거\n
    to() -> type(self) 깊은복사\n
    clone() -> type(self) 깊은복사\n
    init_like() -> type(self) 깊은복사\n
    parameters() -> list\n
    
    to는 init_like() + 파라미터들.detach().clone().requires_grad_()
    clone은 자기 타입(바뀐 device,dtype) + 파라미터들.detach().clone().requires_grad_()
    '''
    def __init__(self, device='cpu', dtype=torch.float32, *init_args, **init_kwargs):
        super().__setattr__("_device"         , torch.device(device))
        super().__setattr__("_dtype"          , dtype)
        super().__setattr__("_buffer"         , {})
        super().__setattr__("_modules"        , {})
        super().__setattr__("_parameter_dicts", {})
        super().__setattr__("_others", {})
        self._device            :torch.device
        self._dtype             :torch.dtype
        self._buffer            :dict[str,torch.Tensor]
        self._modules           :dict[str,BaseModule]
        self._parameter_dicts   :dict[str,dict[str,torch.Tensor|BaseModule]]
        self._parameter_lists   :dict[str,list[torch.Tensor]]
        self._parameters        :dict[str, torch.Tensor]
        self._others            :dict[str, Any]
        
        super().__setattr__("_init_args", init_args)
        super().__setattr__("_init_kwargs", init_kwargs)
        self._init_args      :tuple[Any]
        self._init_kwargs    :dict[str, Any]
        
    def add_to_state(self, name:str, value:Any)->None:
        self._others[name] = value
    
    @overload
    def add_buffer(self, name:str, param:torch.Tensor)->None:...
    @overload
    def add_buffer(self, name:str, module:BaseModule)->None:...
    def add_buffer(self, name:str, param_or_module:torch.Tensor|BaseModule)->None:
        if isinstance(param_or_module, BaseModule):
            self._modules[name] = param_or_module
            for k,v in param_or_module._buffer.items():
                self.add_buffer(f'{name}.{k}', v)
        elif is_parameter(param_or_module):
            self._buffer[name] = param_or_module
    
    def __setattr__(self, name:str, value)->None:
        '''
        
        '''
        if not hasattr(self, '_buffers'):
            raise AttributeError("Must have to call super().__init__ before set parameters or modules")
        
        if isinstance(value, torch.Tensor) and value.requires_grad:
            if not value.is_leaf:
                raise ValueError(f"Parameter {name} in {self.__class__.__name__} is set to requires_gard=True, but not leaf.")
            self.add_buffer(name, value)
        
        if isinstance(value, BaseModule):
            self.add_buffer(name, value)
        
        if isinstance(value, dict):
            with run_on_exit_if_true(lambda:self._parameter_dicts.update({name:value})) as trigger:
                for k,v in value.items():
                    if is_parameter_or_module(v):
                        self.add_buffer(f'{name}[{k}]', v)
                        trigger.true_()
        
        if isinstance(value, list):
            with run_on_exit_if_true(lambda:self._parameter_lists.update({name:value})) as trigger:
                for i,v in enumerate(value):
                    if is_parameter_or_module(v):
                        self.add_buffer(f'{name}[{i}]', v)
                        trigger.true_()
        
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
    def _init_weights_he(self):
        for key,param in self._buffer.items():
            if 'weight' in key:
                d_out = param.size(-2)
                param.mul_(math.sqrt(1/d_out))
    @torch.no_grad()
    def _init_weights_xavier(self):
        with torch.no_grad():
            for key,param in self._buffer.items():
                if 'weight' in key:
                    d_in = param.size(-1)
                    d_out = param.size(-2)
                    param.mul_(math.sqrt(2/(d_in+d_out)))
    @torch.no_grad()
    def _init_weights_fn(self, fn:Callable[[d_in, d_out, parameter_name, parameter], torch.Tensor]):
        with torch.no_grad():
            for parameter_name,parameter  in self._buffer.items():
                d_in = parameter.size(-1)
                d_out = parameter.size(-2)
                parameter.data = fn(d_in, d_out, parameter_name, parameter)

    d_in = NewType("d_in", int)
    d_out = NewType("d_out", int)
    parameter_name = NewType("parameter_name", str)
    parameter = NewType("parameter", torch.Tensor)
    @overload
    def init_weights(self, init_method=None)->None:...
    @overload
    def init_weights(self, init_method:str='none')->None:...
    @overload
    def init_weights(self, init_method_fn:Callable[[d_in, d_out, parameter_name, parameter],Any])->None:...
    
    def init_weights(self, init_method:Optional[str|Callable[[d_in, d_out, parameter_name, parameter], torch.Tensor]]='none') -> None:
        '''

        '''
        if isinstance(init_method, Callable):
            self._init_weights_fn(init_method)
            return
        
        init_method = init_method or 'none'
        init_method = init_method.lower()
        
        if init_method=='he':
            self._init_weights_he()
        elif init_method=='xavier':
            self._init_weights_xavier()
    
    @torch.no_grad()
    def _apply(self, fn:Callable[[torch.Tensor], torch.Tensor])->BaseModule:
        for module in self._modules.values():
            module._apply(fn)
        for dict in self._parameter_dicts.values():
            for param in dict.values():
                param.data = fn(param)
        for param_list in self._parameter_lists.values():
            for param in param_list:
                param.data = fn(param)
        for param in self._parameters.values():
            param.data = fn(param)
        return self
    
    def __call__(self, *args, **kwargs):
        if not hasattr(self,'forward'):
            raise AttributeError(f"[{self.__class__.__name__}] forward() method is required.")
        return self.forward(*args, **kwargs)
    
    def to(self, *args, **kwargs) -> BaseModule:
        return self._apply(lambda t: t.to(*args, **kwargs))
    def cuda(self) -> BaseModule:
        self._device = torch.device('cuda')
        return self._apply(lambda t: t.cuda())
    def cpu(self) -> BaseModule:
        self._device = torch.device('cpu')
        return self._apply(lambda t: t.cpu())
    def half(self) -> BaseModule:
        self._dtype = torch.half
        return self._apply(lambda t: t.half())
    def float(self) -> BaseModule:
        self.dtype = torch.float
        return self._apply(lambda t: t.float())
    
    
    def parameters(self) -> list[torch.Tensor]:
        return list(self._buffer.values())
    def named_parameters(self) -> dict[str, torch.Tensor]:
        return self._buffer
    def modules(self) -> list[BaseModule]:
        return list(self._modules.values())
    def named_modules(self) -> dict[str, BaseModule]:
        return self._modules
    
    @property  
    def device(self) -> torch.device:
        self._device
    @property
    def dtype(self) -> torch.dtype:
        self._dtype
        
    def init_like(self):
        try:
            return self.__class__(*self._init_args, **self._init_kwargs)
        except (TypeError,ValueError) as e:
            raise TypeError(f"[BaseModule of {self.__class__.__name__}] init_like() was called, but no or invalid initialization arguments were provided.\n(init_args:{self._init_args}, init_kwargs:{self._init_kwargs})") from e
    
    DictParams =NewType('DictParams', dict[str, dict[str, torch.Tensor]])
    ListParams =NewType('ListParams', dict[str, dict[int, torch.Tensor]])
    Modules    =NewType('Modules', dict[str, "BaseModule"])
    Params     =NewType('Params', dict[str, torch.Tensor])
    Others     =NewType('Others', dict[str, Any])
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
                if is_parameter(value):
                    state['dicts'].update({
                        name:{ key:{
                                'data':value.cpu(),
                                'grad':value.grad.cpu() if value.grad is not None else None
                        }}
                    })
                elif isinstance(value, BaseModule):
                    state['dicts'].update({name:{key:value.state_dict()}})
        for name,list in self._parameter_lists.items():
            for i,value in enumerate(list):
                if is_parameter(value):
                    state['lists'].update({
                        name:{ i:{
                                'data':value.cpu(),
                                'grad':value.grad.cpu() if value.grad is not None else None
                        }}
                    })
                if isinstance(value, BaseModule):
                    state['lists'].update({name:{i:{value.state_dict()}}})
        for name,param in self._parameters.items():
            state['parameters'].update({
                name:{
                    'data':param.cpu(),
                    'grad':param.grad.cpu() if param.grad is not None else None
                }
            })
        return state
    
    @torch.no_grad()
    def load_state(self, state:StateDict)->None:
        prefix = f'[{self.__class__.__name__}.load_state()] Invalid key '
        def MakeKeyError(position, *keys):
            return KeyError(prefix + ', '.join(map(str, keys)) + f'in {position}')
        
        try:
            for key,module_state in state['modules'].items():
                self._modules[key].load_state[module_state]
        except KeyError as e:
            raise MakeKeyError('_modules', key) from e
        
        try:
            for key,dict_states in state['dicts'].items():
                for k,dict_state in dict_states.items():
                    if is_parameter(self._parameter_dicts[k]):
                        self._parameter_dicts[k].data = restore_param(dict_state['data'], self.device, self.dtype)
                        self._parameter_dicts[k].grad = dict_state['grad'].to(self.device, self.dtype) \
                            if dict_state['grad'] is not None else None
        except KeyError as e:
            raise MakeKeyError('_parameter_dicts', key,k) from e
        
        try:
            for key,list_states in state['lists'].items(): 
                for i,list_state in list_states.items():
                    if is_parameter(self._parameter_lists[key][i]):
                        self._parameter_lists[key][i].data = restore_param(list_state['data'], self.device, self.dtype)
                        self._parameter_lists[key][i].grad = list_state['grad'].to(self.device, self.dtype) \
                            if list_state['grad'] is not None else None
        except KeyError as e:
            raise MakeKeyError('_parameter_lists', key,i) from e
        
        try:
            for key,param_state in state['parameters'].items():
                self._parameters[key].data = restore_param(param_state['data'], self.device, self.dtype)
                self._parameters[key].grad = param_state['grad'].to(self.device, self.dtype) \
                    if param_state['grad'] is not None else None
        except KeyError as e:
            raise MakeKeyError('_parameters', key) from e
        
        for key,other_state in state['others'].items():
            if not key in self.__dict__:
                raise MakeKeyError('others', key)
            self.__dict__[key] = other_state
        