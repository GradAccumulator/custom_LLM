from ..src.setup import (
    load_cpp,
    Tensor,
    torch
)

_computeMV_wrapper = load_cpp('computeMV_wrapper', 'compute_mv')

def computeMV_wrapper(
    params:dict[str, Tensor], 
    m:dict[str, Tensor], 
    v:dict[str, Tensor], 
    t:int, 
    b1:float, 
    b2:float, 
    eps:float, 
    move_device:bool,
) :
    return _computeMV_wrapper.computeMV_wrapper(
        params,
        m,
        v,
        t,
        b1,
        b2,
        eps,
        move_device
    )
    
