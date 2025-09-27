from ..src.setup import (
    load_cpp,
    Tensor,
    torch
)

_dropout = None
load_cpp("dropout", dict=globals(), name='_dropout')

def dropout(x:Tensor, p:float):
    return _dropout.dropout(x, p)