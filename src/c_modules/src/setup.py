from torch.utils.cpp_extension import load
import torch
Tensor = torch.Tensor
from os import path

_computeMV_wrapper = load(
    name='computeMV_wrapper', 
    sources=[path.join(path.dirname(__file__), "device_transfer.cpp")]
)
print("load device_transfer")
_attention = load(
    name="attention", 
    sources=[path.join(path.dirname(__file__), "attention.cpp")]
)
print("load attention")

def causal_masked_self_attention(
    x:Tensor, 
    qkv_weight:Tensor, 
    output_weight:Tensor, 
    num_heads:int, 
    d_head:int, 
    seq_len:int, 
    p_drop:float=0.0,
    qkv_bias:Tensor=None, 
    output_bias:Tensor=None, 
    scale_qk:Tensor=None, 
    cached_causal_mask:Tensor=None
):
    return _attention.causal_masked_self_attention(
        x,
        qkv_weight,
        output_weight,
        num_heads,
        d_head,
        seq_len,
        p_drop,
        qkv_bias if qkv_bias is not None else torch.empty(0),
        output_bias if output_bias is not None else torch.empty(0),
        scale_qk if scale_qk is not None else torch.empty(0),
        cached_causal_mask if cached_causal_mask is not None else torch.empty(0)
    )

def computeMV_wrapper(
    params:dict[str, Tensor], 
    m:dict[str, Tensor], 
    v:dict[str, Tensor], 
    t:int, 
    b1:float, 
    b2:float, 
    eps:float, 
    move_device:bool,
    device:torch.device
) :
    return _computeMV_wrapper.computeMV_wrapper(
        params,
        m,
        v,
        t,
        b1,
        b2,
        eps,
        move_device,
        device
    )