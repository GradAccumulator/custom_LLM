from ..src.setup import (
    load_cpp,
    Tensor,
    torch
)

_dropout = load_cpp("dropout")

def dropout(x:Tensor, p:float):
    return _dropout.dropout(x, p)