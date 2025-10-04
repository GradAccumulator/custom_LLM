import torch
from .base_module import Module
from .linear import Linear
from .scale import Scale
from .activation_functions import Softmax
from .dropout import Dropout
import src.c_modules as _C
import math

class IntRef(int):
    def __init__(self, value):
        self.value = value
        self = value

class SelfAttention(Module):
    def __init__(
        self, 
        d_model,
        num_heads, 
        d_head=None,
        *,
        device='cpu', 
        dtype=torch.float32, 
        use_bias=False, 
        init_method='Xavier', 
        mask_dtype=torch.float16, 
        seq_len=None, 
        qk_std_init:float=None, 
        learnable_qk_std=True, 
        dropout_p=0.1,
        eps:float=1e-4
    ):
        if d_model%num_heads != 0 and d_head is None:
            raise AttributeError("d_model과 head 개수가 맞지 않습니다.")
        super().__init__(device, dtype, d_model=d_model, num_heads=num_heads, d_head=d_head, use_bias=use_bias, init_method=init_method, mask_dtype=mask_dtype, seq_len=seq_len, qk_std_init=qk_std_init, learnable_qk_std=learnable_qk_std, dropout_p=dropout_p, eps=eps)
        if d_head is None:
            self.d_head = d_model//num_heads
        else:
            self.d_head = d_head
        self.d_model = d_model
        self.num_heads = num_heads
        self._use_bias = use_bias
        self.init_method = init_method.lower()
        self.seq_len = seq_len
        self.softmax = Softmax(dim=-1)
        self.use_dropout = dropout_p>0
        self.mask_dtype = mask_dtype
        self.eps = eps
        self._is_inference = False
        if self.use_dropout:
            self.dropout = Dropout(p=dropout_p)
        
        if self.seq_len is not None:
            self.cached_mask = torch.full((self.seq_len, self.seq_len), float('-inf'), device=self._device, dtype=mask_dtype).triu(diagonal=1)
            self.add_to__apply('cached_mask', self.cached_mask)
        else: 
            self.cached_mask = None
            
        self.recip_root_d_k = 1/(self.d_head**0.5)
        self.qk_std_init = qk_std_init

        if self.qk_std_init is not None:
            self.qk_std_exp = Scale(math.exp(self.qk_std_init)-1, shape=(self.num_heads, 1, 1), learnable=learnable_qk_std, device=self._device, dtype=self._dtype)
            self.use_qk_std = True
        else:
            self.use_qk_std = False
        self.reset_params()
        self.train()
    
    def reset_params(self):
        self.linear_qkv = Linear(self.d_model, self.num_heads*self.d_head*3, device=self._device, dtype=self._dtype, init_method=self.init_method, use_bias=self.use_bias)
        self.linear_output = Linear(self.num_heads*self.d_head, self.d_model, device=self._device, dtype=self._dtype, init_method=self.init_method, use_bias=self.use_bias)
    
    def reset_cache(self, set_to_none=False):
        self.cache_offset = IntRef(0)
        if set_to_none:
            self.k_cache = None
            self.v_cache = None
            return
        self.k_cache = torch.empty((self.seq_len, self.num_heads, self.d_head), device=self._device, dtype=self._dtype)
        self.v_cache = torch.empty((self.seq_len, self.num_heads, self.d_head), device=self._device, dtype=self._dtype)
    
    @property
    def use_bias(self):
        return self._use_bias
    def forward(self, x:torch.Tensor):
        '''
        input shape:(batch, seq_len, d_model)\n
        output shape:(batch, seq_len, d_model)
        '''
        return _C.causal_masked_self_attention(
            x,
            self.linear_qkv.weight.T,
            self.linear_output.weight.T,
            self.num_heads,
            self.d_head,
            self.seq_len,
            self.dropout.p if self.use_dropout and self.dropout.is_training else 0.0,
            self.linear_qkv.bias if self.use_bias else None,
            self.linear_output.bias if self.use_bias else None,
            self.qk_std_exp.scale if self.use_qk_std else None,
            self.cached_mask if self.cached_mask is not None else None,
            self.eps,
            self.k_cache,
            self.v_cache,
            self.is_inferencing,
            self.cache_offset
        )
    
    def train(self):
        self.reset_cache(set_to_none=True)
        return super().train()
    def eval(self):
        self.reset_cache(set_to_none=True)
        return super().eval()
    def inference(self):
        self.reset_cache()
        return super().inference()
        
    def __repr__(self):
        s_goal_exp_repr = ''
        if hasattr(self, 's_goal_exp'):
            s_goal_exp_repr = (
                f'    s_goal_exp_value={self.qk_std_init}\n'
                f'    learnable_s_goal_exp={self.qk_std_exp.learnable}\n'
            )
            if self.qk_std_exp.learnable:
                s_goal_exp_repr += (
                    f"    <<qk_std_exp>>\n"
                    f'    {repr(self.qk_std_exp.scale.flatten())}\n'
                )
        return (
            f'SelfAttention(\n'
            f'    d_model={self.d_model},\n'
            f'    num_heads={self.num_heads},\n'
            f'    device={self._device},\n'
            f'    dtype={self._dtype},\n'
            f'    init_method={self.init_method},\n'
            f'    seq_len={self.seq_len},\n'
            f'    dropout_p={self.dropout.p if self.use_dropout else 0.0},\n'
            f'    use_bias={self.use_bias},\n'
            +s_goal_exp_repr+
            f'    mask_dtype={self.mask_dtype}\n'
            f')'
        )