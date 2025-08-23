import torch
from .base_module import BaseModule
from .linear import Linear
from .scale import Scale
from .activation_functions import Softmax
from .dropout import Dropout

class SelfAttention(BaseModule):
    def __init__(self, d_model, num_heads,*,device='cpu', dtype=torch.float32, use_bias=False, init_method='Xavier', mask_type='causal', seq_len=None, scale_qk_value=None, learnable_scale_qk=False, dropout_p=0.1):
        if d_model%num_heads != 0:
            raise AttributeError("d_model과 head 개수가 맞지 않습니다.")
        super().__init__(device, dtype)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model//num_heads
        self._use_bias = use_bias
        self.init_method = init_method.lower()
        self.seq_len = seq_len
        self.mask_type = mask_type
        self.softmax = Softmax(dim=-1, dtype=self.dtype)
        self.dropout = dropout_p>0
        if self.dropout:
            self.dropout = Dropout(dropout_p)
        
        if self.mask_type=='causal' and self.seq_len is not None:
            self.cached_mask = torch.full((self.seq_len, self.seq_len), float('-inf'), device=self.device, dtype=self.dtype)
            self.cached_mask = self.cached_mask.triu(diagonal=1)
            
        self.recip_root_d_k = 1/(self.d_head**0.5)
        self.scale_qk_value = scale_qk_value
        
        if self.scale_qk_value is not None:
            self.scale_qk = Scale(self.scale_qk_value, shape=(self.num_heads, 1, 1), learnable=learnable_scale_qk, device=self.device, dtype=self.dtype)
        self.reset_params()
    
    def reset_params(self):
        self.linear = Linear(self.d_model, self.num_heads*self.d_head*3, device=self.device, dtype=self.dtype, init_method=self.init_method, use_bias=self.use_bias)
    
    @property
    def use_bias(self):
        return self._use_bias
    
    def _make_mask(self, seq_len):
        if self.seq_len == seq_len:
            return self.cached_mask
        return torch.full((seq_len, seq_len), float('-inf'), device=self.device, dtype=self.dtype).triu(diagonal=1)
    
    def _get_qkv(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        input shape:(B, seq_len, d_model)
        output shape:(B, num_heads, seq_len, d_head), (Q,K,V)
        '''
        Q,K,V = self.linear(x).view(x.shape[0], x.shape[1], self.num_heads, self.d_head*3).transpose(1, 2).split(self.d_head, dim=-1)
        if self.scale_qk_value is not None:
            return (Q*self.scale_qk, K*self.scale_qk,V)
        return (Q,K,V)
    
    def _attention(self, Q,K,V, seq_len):
        scores = (Q@K.transpose(-2, -1))*self.recip_root_d_k
        if seq_len == self.seq_len:
            masked = scores + self.cached_mask
        else:
            masked = scores + self._make_mask(seq_len)
        weight = self.softmax(masked)
        if self.dropout:
            weight = self.dropout(weight)
        return (weight@V)
    
    def forward(self, x:torch.Tensor):
        '''
        input shape:(batch, seq_len, d_model)\n
        output shape:(batch, seq_len, d_model)
        '''
        Q,K,V = self._get_qkv(x)
        atten_out = self._attention(Q, K, V, x.shape[1]).transpose(1,2)
        return atten_out
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return self.linear.parameters()
    
    def to(self, device=None, dtype=None):
        new = self.init_like()
        new.linear = self.linear.to(device, dtype)
        if self.scale_qk_value is not None:
            new.scale_qk = self.scale_qk.to(device, dtype)
        return new
    
    def clone(self):
        atten = SelfAttention(self.d_model, self.num_heads, device=self.device, dtype=self.dtype, use_bias=self.use_bias, init_method=self.init_method, mask_type=self.mask_type, seq_len=self.seq_len)
        atten.linear = self.linear.clone()
    
    def init_like(self):
        return SelfAttention(self.d_model, self.num_heads, device=self.device, dtype=self.dtype, use_bias=self.use_bias, init_method=self.init_method, mask_type=self.mask_type, seq_len=self.seq_len, scale_qk_value=self.scale_qk_value, learnable_scale_qk=self.scale_qk.learnable, dropout_p=self.dropout.p if self.dropout else 0.0)
    
    def state_dict(self):
        return {
            'linear': self.linear.state_dict(),
            'scale_qk': self.scale_qk.state_dict(),
        }
        
    def load_state(self, state):
        try:
            self.linear.load_state_dict(state['linear'])
            self.scale_qk.load_state_dict(state['scale_qk'])
        except ValueError as e:
            print("<< SelfAttention load_state error >>")
            raise e