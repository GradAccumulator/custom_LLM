from torch.utils.cpp_extension import load
import torch
Tensor = torch.Tensor
from os import path
from src.utils import threads

def _load(func_name, dict, name, file_name:str|None=None):
    try:
        dict[name] = load(
            name=func_name, 
            sources=[path.join(path.dirname(__file__), (file_name)+'.cpp')]
        )
    except RuntimeError as e:
        if e.args[0].startswith('Error building extension'):
            _load(func_name, dict, name, file_name)

def load_cpp(func_name, file_name:str|None=None,*,dict,name):
    if file_name is None:
        file_name = func_name
    wrapper = threads.start(target=_load, func_name=func_name, file_name=file_name,dict=dict, name=name)
    print(f'[loading] {file_name+".cpp"}/{func_name}')

__all__ = [
    "Tensor",
    "torch",
    "load_cpp"
]