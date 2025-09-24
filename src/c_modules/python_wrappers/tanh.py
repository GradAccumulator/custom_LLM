from ..src.setup import (
    load_cpp,
    Tensor,
    torch
)

_tanh = load_cpp("tanh")

def tanh(x:Tensor):
    return _tanh.tanh(x)