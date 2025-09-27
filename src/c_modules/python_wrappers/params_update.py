from ..src.setup import *


_step_with_wd   = None
_step_with_none = None
_zero_grad      = None
load_cpp('step_with_wd', 'params_update', dict=globals(), name="_step_with_wd")
load_cpp('step_with_none', 'params_update', dict=globals(), name="_step_with_none")
load_cpp("zero_grad", "params_update", dict=globals(), name="_zero_grad")

def step_with_wd(
    map_tensor:dict[str, dict[str, Tensor|tuple[Tensor, float]]],
    gradients:dict[str, Tensor],
    lr,
    wd
):
    _step_with_wd.step_with_wd(
        map_tensor,
        gradients,
        lr,
        wd
    )

def step_with_none(
    map_tensor:dict[str, dict[str, Tensor|tuple[Tensor, float]]],
    gradients:dict[str, Tensor],
    lr
):
    _step_with_none.step_with_none(
        map_tensor,
        gradients,
        lr
    )
def zero_grad(params:list[Tensor]):
    _zero_grad.zero_grad(params)