from ..src.setup import (
    load_cpp,
    Tensor,
    torch
)

_attention = None
load_cpp('attention', dict=globals(), name='_attention')

class IntRef(int):
    def __init__(self, value):
        self.value = value
        self = value

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
    cached_causal_mask:Tensor=None,
    eps:float=1e-5,
    k_cache=None,
    v_cache=None,
    is_inference=False,
    cache_offset:IntRef=IntRef(0)
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
        cached_causal_mask if cached_causal_mask is not None else torch.empty(0),
        eps,
        k_cache if k_cache is not None else torch.empty(0),
        v_cache if v_cache is not None else torch.empty(0),
        is_inference,
        cache_offset.value if cache_offset is not None else 0
    )
