import torch
from .base_module import BaseModule
from .linear import Linear
from .scale import Scale
from .activation_functions import Softmax
from .dropout import Dropout
from src.utils.torch_utils import device_dtype_check

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
        mask_type='causal', 
        mask_dtype=torch.float16, 
        seq_len=None, 
        scale_qk_value=None, 
        learnable_scale_qk=False, 
        dropout_p=0.1
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
        self.mask_type = mask_type
        self.softmax = Softmax(dim=-1)
        self.use_dropout = dropout_p>0
        self.mask_dtype = mask_dtype
        if self.use_dropout:
            self.dropout = Dropout(dropout_p)
        
        if self.mask_type=='causal' and self.seq_len is not None:
            self.cached_mask = torch.full((self.seq_len, self.seq_len), float('-inf'), device=self.device, dtype=mask_dtype).triu(diagonal=1)
            
        self.recip_root_d_k = 1/(self.d_head**0.5)
        self.scale_qk_value = scale_qk_value

        if self.scale_qk_value is not None:
            self.scale_qk = Scale(self.scale_qk_value, shape=(self.num_heads, 1, 1), learnable=learnable_scale_qk, device=self.device, dtype=self.dtype)
        self.use_scale = self.scale_qk_value is not None
        self.reset_params()
    
    def reset_params(self):
        self.linear_qkv = Linear(self.d_model, self.num_heads*self.d_head*3, device=self.device, dtype=self.dtype, init_method=self.init_method, use_bias=self.use_bias)
        self.linear_output = Linear(self.num_heads*self.d_head, self.d_model, device=self.device, dtype=self.dtype, init_method=self.init_method, use_bias=self.use_bias)
    
    @property
    def use_bias(self):
        return self._use_bias
    
    def _make_mask(self, seq_len, dtype=torch.float32):
        if self.seq_len == seq_len:
            return self.cached_mask
        return torch.full((seq_len, seq_len), float('-inf'), device=self.device, dtype=dtype).triu(diagonal=1)
    
    def _get_qkv(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        input shape:(B, seq_len, d_model)
        output shape:(B, num_heads, seq_len, d_head), (Q,K,V)
        '''
        Q,K,V = self.linear_qkv(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_head*3).transpose(1, 2).split(self.d_head, dim=-1)
        if self.use_scale:
            return (Q*self.scale_qk, K*self.scale_qk,V)
        return (Q,K,V)
    
    def _attention(self, Q,K,V, seq_len):
        scores = (Q@K.transpose(-2, -1))*self.recip_root_d_k
        if seq_len == self.seq_len:
            masked = scores + self.cached_mask.to(scores.dtype)
        else:
            masked = scores + self._make_mask(seq_len, dtype=scores.dtype)
        weight = self.softmax(masked)
        if self.use_dropout:
            weight = self.dropout(weight)
        return (weight@V)
    
    def forward(self, x:torch.Tensor):
        '''
        input shape:(batch, seq_len, d_model)\n
        output shape:(batch, seq_len, d_model)
        '''
        Q,K,V = self._get_qkv(x)
        atten_out = self._attention(Q, K, V, x.shape[1]).transpose(1,2).reshape(x.shape[0], x.shape[1], self.num_heads*self.d_head)
        out = self.linear_output(atten_out)
        return out
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device=None, dtype=None):
        device, dtype = device_dtype_check(device, dtype)
        new = self.init_like()
        device = device or self.device
        dtype = dtype or self.device
        new.linear_qkv = self.linear_qkv.to(device, dtype)
        new.linear_output = self.linear_output.to(device, dtype)
        new.cached_mask = self.cached_mask.to(device, dtype)
        if self.scale_qk_value is not None:
            new.scale_qk = self.scale_qk.to(device, dtype)
        new.device = device
        new.dtype = dtype
        return new
    
    def eval(self):
        self.dropout.eval()
    def train(self):
        self.dropout.train()
    @property
    def mode(self):
        return self.dropout.mode
    
    def clone(self):
        atten = self.init_like()
        atten.linear_qkv = self.linear_qkv.clone()
        atten.linear_output = self.linear_output.clone()
        if self.use_scale:
            atten.scale_qk = self.scale_qk.clone()
        return atten
    
    def init_like(self):
        return SelfAttention(
            self.d_model, self.num_heads, 
            device              =self.device, 
            dtype               =self.dtype, 
            use_bias            =self.use_bias, 
            init_method         =self.init_method, 
            mask_type           =self.mask_type, 
            seq_len             =self.seq_len, 
            scale_qk_value      =self.scale_qk_value, 
            learnable_scale_qk  =hasattr(self, 'scale_qk') and self.scale_qk.learnable, 
            dropout_p           =self.dropout.p if self.use_dropout else 0.0,
            mask_dtype          =self.mask_dtype
        )
    
    def state_dict(self):
        return {
            'linear': self.linear_qkv.state_dict(),
            'scale_qk': self.scale_qk.state_dict() if hasattr(self, 'scale_qk') and self.scale_qk.learnable else None,
            'W_o': self.linear_output.state_dict(),
        }
        
    def load_state(self, state):
        try:
            self.linear_qkv.load_state(state['linear'])
            if hasattr(self, 'scale_qk') and self.scale_qk.learnable:
                self.scale_qk.load_state(state['scale_qk'])
        except ValueError as e:
            print("Error in SelfAttention.load_state")
            raise e
        
    def __repr__(self):
        scale_qk_repr = ''
        if hasattr(self, 'scale_qk'):
            scale_qk_repr = (
                f'scale_qk_value={self.scale_qk_value}\n'
                f'learnable_scale_qk={self.scale_qk.learnable}\n'
            )
            if self.scale_qk.learnable:
                scale_qk_repr += (
                    f"<<scale_qk>>\n"
                    f'{repr(self.scale_qk.scale.flatten())}\n'
                )
        return (
            f'SelfAttention(\n'
            f'    d_model={self.d_model},\n'
            f'    num_heads={self.num_heads},\n'
            f'    device={self.device},\n'
            f'    dtype={self.dtype},\n'
            f'    init_method={self.init_method},\n'
            f'    mask_type={self.mask_type},\n'
            f'    seq_len={self.seq_len},\n'
            f'    dropout_p={self.dropout.p if self.use_dropout else 0.0},\n'
            +scale_qk_repr+
            f'    mask_dtype={self.mask_dtype}\n'
            f')'
        )