from torch.utils.cpp_extension import load
import torch
Tensor = torch.Tensor
from os import path

def load_cpp(func_name, file_name:str|None=None):
    if file_name is None:
        file_name = func_name
    wrapper = load(
        name=func_name,
        sources=[path.join(path.dirname(__file__), (file_name)+'.cpp')]
    )
    print(f'[loaded] {file_name+".cpp"}/{func_name}')
    return wrapper

__all__ = [
    "Tensor",
    "torch",
    "load_cpp"
]