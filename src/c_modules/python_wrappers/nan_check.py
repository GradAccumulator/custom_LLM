from ..src.setup import (
    load_cpp,
    Tensor,
    torch
)

_nan_check = None
_grad_nan_check = None
load_cpp('nan_check', 'nan_check', dict=globals(), name='_nan_check')
load_cpp('grad_nan_check', 'nan_check', dict=globals(), name='_grad_nan_check')

def nan_check(tensor_list :list[Tensor]) -> bool:
    return _nan_check.nan_check(tensor_list)
def grad_nan_check(tensor_list:list[Tensor]) -> bool:
    return _grad_nan_check.grad_nan_check(tensor_list)