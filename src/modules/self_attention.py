import torch
from .base_module import BaseModule
from .linear import Linear
from .scale import Scale
from .activation_functions import Softmax
from .dropout import Dropout
from src.utils.torch_utils import 보이면_바로_교체하기
import src.c_modules as _C
import math

class IntRef(int):
    def __init__(self, value):
        self.value = value
        self = value

class SelfAttention(BaseModule):
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
        if d_head is None:
            self.d_head = d_model//num_heads
        else:
            self.d_head = d_head
        super().__init__(device, dtype)
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
    
    def reset_cache(self):
        self.k_cache = torch.empty((self.seq_len, self.num_heads, self.d_head), device=self._device, dtype=self._dtype)
        self.v_cache = torch.empty((self.seq_len, self.num_heads, self.d_head), device=self._device, dtype=self._dtype)
        self.cache_offset = IntRef(0)
    @property
    def is_training(self):
        return not self._is_inference
    @property
    def mode(self):
        return 'train' if self.is_training else 'inference'
    @property
    def is_inference(self):
        return self._is_inference
    
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
            self.is_inference,
            self.cache_offset
        )
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device=None, dtype=None):
        device, dtype = 보이면_바로_교체하기(device, dtype)
        new = self.init_like()
        device = device or self._device
        dtype = dtype or self._dtype
        new.linear_qkv = self.linear_qkv.to(device, dtype)
        new.linear_output = self.linear_output.to(device, dtype)
        new.cached_mask = self.cached_mask.to(device, dtype)
        if self.qk_std_init is not None:
            new.qk_std_exp = self.qk_std_exp.to(device, dtype)
        new._device = device
        new._dtype = dtype
        return new
    
    def eval(self):
        if self.use_dropout:
            self.dropout.eval()
    def train(self):
        if self.use_dropout:
            self.dropout.train()
        self._is_inference = False
        self.k_cache = None
        self.v_cache = None
        self.cache_offset = IntRef(0)
    def inference(self):
        self._is_inference = True
        self.reset_cache()
    @property
    def mode(self):
        return self.dropout.mode
    
    def clone(self):
        atten = self.init_like()
        atten.linear_qkv = self.linear_qkv.clone()
        atten.linear_output = self.linear_output.clone()
        if self.use_qk_std:
            atten.qk_std_exp = self.qk_std_exp.clone()
        return atten
    
    def init_like(self):
        return SelfAttention(
            self.d_model, self.num_heads, 
            device              =self._device, 
            dtype               =self._dtype, 
            use_bias            =self.use_bias, 
            init_method         =self.init_method, 
            seq_len             =self.seq_len, 
            qk_std_init         =self.qk_std_init, 
            learnable_qk_std    =hasattr(self, 'qk_std_exp') and self.qk_std_exp.learnable, 
            dropout_p           =self.dropout.p if self.use_dropout else 0.0,
            mask_dtype          =self.mask_dtype
        )
    
    def state_dict(self):
        return {
            'linear': self.linear_qkv.state_dict(),
            'qk_std_exp': self.qk_std_exp.state_dict() if hasattr(self, 'qk_std_exp') and self.qk_std_exp.learnable else None,
            'W_o': self.linear_output.state_dict(),
        }
        
    def load_state(self, state):
        try:
            self.linear_qkv.load_state(state['linear'])
            if hasattr(self, 's_goal_exp') and self.qk_std_exp.learnable:
                self.qk_std_exp.load_state(state['s_goal_exp'])
        except ValueError as e:
            print("Error in SelfAttention.load_state")
            raise e
        
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
        return super().__repr__((
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
        ))