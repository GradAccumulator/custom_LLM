from __future__ import annotations
import torch
import math
#from torch.nn.functional import scaled_dot_product_attention
from ...nlp.tokenizer import koTokenizer
from ...modules.activation_functions import Softmax
from ...nlp.embedding import Embedding

def make_L_triangle_matrix(v, h, device, dtype) -> torch.Tensor:
    res = torch.zeros(v, h, device=device, dtype=dtype)
    for i in range(v):
        res[i,i+1:] = float('-inf')
    return res

class Transformer:
    @staticmethod
    def load_state(tokenizer, act_fn, state:dict):
        transformer = Transformer(
            seq_len     =state['seq_len'],
            num_heads   =state['num_heads'],
            d_model     =state['d_model'],
            num_layers  =state['num_layers'],
            d_ffn       =state['d_ffn'],
            act_fn      =act_fn,
            p_dropout   =state['p'],
            device      =state['device'],
            LN_eps      =state['LN_eps'],
            tokenizer   =tokenizer,
            dtype       =state['dtype'],
            skip_reset  =True
        )
        transformer.params                      = state['params']
        transformer.embedding                   = Embedding.load_state(state['embedding'])
        transformer.pos_embedding               = Embedding.load_state(state['pos_embedding'])
        transformer.params['token_embedding']   = transformer.embedding.table
        transformer.params['pos_embedding']     = transformer.pos_embedding.table
        
        for k,p in transformer.params.items():
            if k in ['token_embedding', 'pos_embedding']:
                continue
            transformer.params[k] = p.detach().to(state['device']).requires_grad_()

        # for key in transformer.params.keys():
        #     if key in ['token_embedding', 'pos_embedding']:
        #         continue
        #     transformer.params[key] = transformer.params[key].detach().to(state['device']).requires_grad_()

        return transformer

    def state_dict(self):
        params = {}
        for key in self.params.keys():
            if key in ['token_embedding', 'pos_embedding']:
                continue
            params[key] = self.params[key].cpu().detach().clone()
            
        state = {
            'p':self._p,
            'dtype':self.dtype,
            'd_ffn':self.d_ffn,
            'device':self.device,
            'LN_eps':self.LN_eps,
            'd_model':self.d_model,
            'seq_len':self.seq_len,
            'num_heads':self.num_heads,
            'num_layers':self.num_layers,
            'd_head':self.d_head,
            'params':self.params,
            'embedding':self.embedding.state_dict(),
            'pos_embedding':self.pos_embedding.state_dict()
        }
        return state

    def _init_hyperparameters(self, seq_len, num_heads, d_model, LN_eps, num_layers, d_ffn, act_fn,device, p, dtype, scale_qk, scale_res_ffn, scale_res_atten):
        #받아온거를 그대로 넣는 객체 변수들
        self._p   = p
        self.t   = 1
        self.dtype  = dtype
        self.d_ffn  = d_ffn
        self.device = device
        self.LN_eps = LN_eps
        self.d_model   = d_model
        self.seq_len   = seq_len
        self.num_heads = num_heads
        self.num_layers   = num_layers
        self.act_fn       = act_fn
        self.d_head       = d_model//num_heads
        self.vocab_size   = self.tokenizer.vocab_size
        self.scale_qk     = scale_qk
        self.scale_res_ffn = scale_res_ffn
        self.scale_res_atten = scale_res_atten

        #받아온거를 이욯해서 대입하는 객체 변수들
        self.wkqv_shape  = (self.num_layers, self.d_model,self.num_heads*self.d_head*3)
        self.causal_mask = make_L_triangle_matrix(seq_len, seq_len, self.device, torch.float16)
        self.root_d_k    = torch.sqrt(torch.tensor(self.d_head, device=self.device, dtype=torch.float16))
        self.recip_root_d_k = 1/self.root_d_k
        self.recip_one_minus_p = 1/(1-self._p)

        self.std_ffn_weights_in = math.sqrt(1/self.d_ffn)
        self.std_ffn_weights_out  = math.sqrt(2/(self.d_ffn+self.d_model))
        self.std_Wkqv  = math.sqrt(2/(self.d_model+self.d_model))
        self.std_Wo    = math.sqrt(2/(self.d_model+self.vocab_size))
        self.std_atten_Wo = math.sqrt(2/(self.num_heads*self.d_head+self.d_model))
        self.arranged  = torch.arange(self.seq_len, device=self.device)

        #when use RoPE
        '''k = torch.arange(0, self.d_head/2, device=self.device)
        self.thetas = 10000(-2*k/self.d_head)
        p = torch.arange(self.seq_len, device=self.device)
        self.p_thetas = p.unsqueeze(1)*self.thetas.unsqueeze(0)'''
    @property
    def p(self):
        return self._p
    @p.setter
    def p(self, p):
        self._p = p
        self.recip_one_minus_p = 1/(1-self._p)

    def _reset_params(self):
        self.params = {
            'qkv_weights': (torch.randn(self.wkqv_shape, device=self.device, dtype=torch.float32)*self.std_Wkqv).requires_grad_(),
            
            'atten_output_weights': (torch.randn(self.num_heads*self.d_head, self.d_model, device=self.device, dtype=torch.float32)*self.std_atten_Wo).requires_grad_(),

            'ffn_weights_in' : (torch.randn(self.num_layers, self.d_model, self.d_ffn, device=self.device, dtype=torch.float32)*self.std_ffn_weights_in).requires_grad_(),

            'ffn_biases_in' : torch.zeros(self.num_layers, self.d_ffn, device=self.device, dtype=torch.float32, requires_grad=True),

            'ffn_weights_out' : (torch.randn(self.num_layers, self.d_ffn, self.d_model, device=self.device, dtype=torch.float32)*self.std_ffn_weights_out).requires_grad_(),

            'ffn_biases_out' : torch.zeros(self.num_layers, self.d_model, device=self.device, dtype=torch.float32, requires_grad=True), 

            'attention_gamma': torch.ones(self.num_layers, self.d_model, device=self.device, dtype=torch.float32, requires_grad=True),

            'attention_shift': torch.zeros(self.num_layers, self.d_model, device=self.device, dtype=torch.float32, requires_grad=True),

            'ffn_gamma'  : torch.ones(self.num_layers, self.d_model, device=self.device, dtype=torch.float32, requires_grad=True),

            'ffn_shift'  : torch.zeros(self.num_layers, self.d_model, device=self.device, dtype=torch.float32, requires_grad=True),

            #'lm_gamma'   : torch.ones(self.d_model, device=self.device, dtype=torch.float32, requires_grad=True),

            #'lm_shift'   : torch.zeros(self.d_model, device=self.device, dtype=torch.float32, requires_grad=True),

            'scale_qk'   : torch.full((self.num_layers, self.num_heads, 1, 1), self.scale_qk,device=self.device, dtype=torch.float32, requires_grad=True),

            'scale_res_ffn'  : torch.full((self.num_layers,), self.scale_res_ffn, device=self.device, dtype=torch.float32, requires_grad=True),

            'scale_res_atten'  : torch.full((self.num_layers,), self.scale_res_atten, device=self.device, dtype=torch.float32, requires_grad=True),

            #'lm_bias'    : torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32, requires_grad=True),

            #'scale_lm'   : torch.tensor(0.2, device=self.device, dtype=torch.float32, requires_grad=True),

            'token_embedding': self.embedding.table,
            'pos_embedding'  : self.pos_embedding.table,
        }
    
    def parameters(self)-> list[torch.Tensor]:
        return list(self.params.values())
    def parameters_as_dict(self)-> dict[str,torch.Tensor]:
        return self.params

    def __init__(self, seq_len:int, num_heads:int, d_model:int, num_layers:int, d_ffn:int, act_fn, tokenizer:koTokenizer, p_dropout=0.0, device='cuda',LN_eps=1e-5, dtype:torch.dtype=torch.float32, scale_qk=0.5, scale_res_ffn=1.0, scale_res_atten=1.0, skip_reset=False):
        if (d_model%num_heads) != 0:
            raise ValueError("d_model과 d_heads가 맞지 않음")

        self.tokenizer = tokenizer

        self._init_hyperparameters(seq_len, num_heads, d_model, LN_eps, num_layers, d_ffn, act_fn,device, p_dropout, dtype, scale_qk, scale_res_ffn, scale_res_atten)

        self.embedding     = Embedding(self.vocab_size, self.d_model, device=self.device, dtype=torch.float32)
        self.pos_embedding = Embedding(self.seq_len, self.d_model, device=self.device, dtype=torch.float32)
        #-------------------------
        if not skip_reset:
            self._reset_params()
        #-------------------------
        self.init_cache()
        self.softmax = Softmax(dim=-1)

    def get_qkv(self, x:torch.Tensor, i:int, batch:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''\n
        ```\n
        input : (B, seq_len, d_model)            
        return (Q, K, V)                        
            └> QKV : (B, num_heads, seq_len, d_head) 
        ```
        '''
        Q,K,V = (x@self.half_params['qkv_weights'][i]).view(batch, self.seq_len, self.num_heads, self.d_head*3).transpose(1,2).split(self.d_head, dim=-1)
        return (Q*self.half_params["scale_qk"][i], K*self.half_params["scale_qk"][i], V)
    
    def init_cache(self):
        '''init KV cache'''
        self.cache_offset = 0
        self.cache = [(
            torch.zeros(1, self.num_heads, self.seq_len, self.d_head, device=self.device, dtype=self.dtype),
            torch.zeros(1, self.num_heads, self.seq_len, self.d_head, device=self.device, dtype=self.dtype),
        ) for _ in range(self.num_layers)] 

    def eval(self):
        #self.init_cache()
        self.mode = 'eval'
        self.reset_half_params()
    def train(self):
        #self.init_cache()
        self.mode = 'train'

    def atten_mask(self, scores:torch.Tensor):
        '''scores:(B, num_heads, seq_len, seq_len)'''
        return scores + self.causal_mask

    def attention(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, dropout:float|bool=False) -> torch.Tensor:
        '''\n
        input : Q,K,V                                     \n
                └>(B, num_heads, seq_len, d_heads)    \n
        return weights@V                                   \n
                └>(B, num_heads, seq_len, d_heads)
        '''

        #Q = self.RoPE(Q)
        #K = self.RoPE(K)
        scores = (Q@K.transpose(-2, -1))*self.recip_root_d_k
        masked = self.atten_mask(scores)
        #self.debug("atten_scores", scores)
        weights = self.softmax(masked)
        if dropout:
            weights = self.dropout(weights)
        
        atten_out = (weights@V).transpose(1,2).reshape(Q.shape[0], Q.shape[2], self.num_heads*self.d_head)
        out = atten_out@self.half_params['atten_output_weights']
        return out
    
    def RoPE(self, x, seq_len=None):
        '''if seq_len is not None:
            p = torch.arange(0,seq_len, device=self.device)
            p_thetas = p.unsqueeze(1)*self.thetas.unsqueeze(0)
        else:
        '''
        p_thetas = self.p_thetas

        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        x_e = x_even*torch.cos(p_thetas) - x_odd*torch.sin(p_thetas)
        x_o = x_even*torch.sin(p_thetas) + x_odd*torch.cos(p_thetas)

        out = torch.empty_like(x)
        out[...,0::2] = x_e
        out[...,1::2] = x_o
        return out

    def residual_atten(self, x:torch.Tensor, atten_out:torch.Tensor,batch:int) -> torch.Tensor:
        '''
        input : (B, num_heads, seq_len, d_heads)
        '''
        atten_out = atten_out.transpose(1, 2).reshape(batch, self.seq_len, self.d_model)
        return x + atten_out
    
    def LN(self, layer:torch.Tensor, gamma:torch.Tensor=None, shift:torch.Tensor=None) -> torch.Tensor:
        mean = layer.mean(dim=-1, keepdim=True)
        var = layer.var(dim=-1, keepdim=True, unbiased=False)
        out = (layer-mean)/(var+self.LN_eps).sqrt()
        if gamma is not None:
            out = gamma*out
        if shift is not None:
            out = out+shift
        return out
    
    def LN_attention(self, s:torch.Tensor, i:int) -> torch.Tensor:
        '''s:(B, seq_len, d_model)'''
        ln_out = self.LN(s, self.half_params['attention_gamma'][i], self.half_params['attention_shift'][i])
        return ln_out
    def LN_ffn(self, s:torch.Tensor, i:int) -> torch.Tensor:
        '''s:(B, seq_len, d_model)'''
        ln_out = self.LN(s, self.half_params['ffn_gamma'][i], self.half_params['ffn_shift'][i])
        return ln_out
    def LN_LanguageModeling_head(self, s:torch.Tensor) -> torch.Tensor:
        '''s:(B, seq_len, d_model)'''
        ln_out = self.LN(s, self.half_params['lm_gamma'], self.half_params['lm_shift'])
        return ln_out

    def FFN(self, x:torch.Tensor, i:int) -> torch.Tensor:
        '''```
        x:(B, seq_len, num_heads*d_head)\n
        output:(B, seq_len, d_model)
        ```'''
        hidden = self.act_fn(x@self.half_params["ffn_weights_in"][i] + self.half_params['ffn_biases_in'][i])
        output = hidden@self.half_params['ffn_weights_out'][i] + self.half_params['ffn_biases_out'][i]
        return output
    
    def LM_head(self, x:torch.Tensor) -> torch.Tensor:
        linear_out = x@self.half_params['token_embedding'].T#+self.half_params['lm_bias']
        return linear_out
    
    def reset_half_params(self):
        self.half_params = {k:v.half() for k,v in self.params.items() if k != 'pos_embedding'}

    def get_norm(self, name:str, i:int, p:int=2) -> float:
        return self.half_params[name][i].norm(p=p)

    def forward_train(self, tokens:torch.Tensor, dropout:None|bool=None):
        '''x:(B, seq_len) 토크나이징 돼있어야함'''
        if dropout is None:
            dropout = self._p>0
        self.reset_half_params()
        batch = len(tokens)

        x = self.embedding(tokens).half()
        x = x + self.pos_embedding(self.arranged).half()
        if dropout: x = self.dropout(x)
        
        for i in range(self.num_layers):
            ln1_out = self.LN_attention(x, i)
            Q,K,V = self.get_qkv(ln1_out, i, batch)
            atten_out = self.attention(Q, K, V, dropout)
            
            if dropout: atten_out = self.dropout(atten_out)
            res_atten = self.residual_atten(x, self.half_params["scale_res_atten"][i]*atten_out, batch)
            
            ln2_out = self.LN_ffn(res_atten, i)
            ffn_out = self.FFN(ln2_out, i)
            
            if dropout: ffn_out = self.dropout(ffn_out)
            x = res_atten + self.half_params["scale_res_ffn"][i]*ffn_out
        logits = self.LM_head(x)

        return logits
    
    def forawrd_eval(self, tokens:torch.Tensor):
        '''
        tokens shape is (B, seq_len) or (seq_len,)   \n
        output shape is (B, seq_len, vocab_size)
        '''
        
        batch = len(tokens)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
            batch = 1
        seq_len_temp = self.seq_len
        self.seq_len = tokens.shape[1]
        
        x = self.embedding(tokens).half()
        x = x + self.pos_embedding(self.arranged).half()
        
        for i in range(self.num_layers):
            ln1_out = self.LN_attention(x, i)
            Q,K,V = self.get_qkv(ln1_out, i, batch)
            atten_out = self.attention(Q, K, V, dropout=False)
            res_atten = self.residual_atten(x, atten_out*self.half_params['scale_res_atten'][i])
            ln2_out = self.LN_ffn(res_atten, i)
            ffn_out = self.FFN(ln2_out, i)
            x = res_atten + ffn_out*self.half_params['scale_res_ffn'][i]
        
        logits = self.LM_head(x)
        self.seq_len = seq_len_temp
        return logits
    
    def forward(self, x:list[str]|torch.Tensor):
        return self.__call__(x)
    
    def __call__(self, x:list[str]|torch.Tensor):
        if self.mode == 'train':
            return self.forward_train(x)
        with torch.no_grad():
            return self.forward_train(x, dropout=False)
    
    def decode_pred(self, x:torch.Tensor):
        '''x는 softmax거친 확률값'''
        return self.tokenizer.decode(x.argmax(dim=-1).tolist())
    
    def dropout(self, nodes:torch.Tensor):
        mask = (torch.rand_like(nodes) >= self._p)
        return nodes * mask*self.recip_one_minus_p

    def decode(self, x:torch.Tensor):
        return self.tokenizer.decode(x)
    
    def encode(self, x:list[str],*, is_dataset=False, max_length=10000) -> torch.Tensor:
        if not is_dataset:
            return torch.tensor(self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=False, max_length=max_length), device=self.device)
        input,label = self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=True, max_length=max_length)
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
        return (input, label)
    def encode_as_list(self, x:list[str], is_dataset=False) -> list[int]:
        return self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=is_dataset)

    @staticmethod
    def debug(*args:str|torch.Tensor):
        log = ''
        for x in args:
            if type(x)==str:
                log = x
            elif torch.isnan(x).any():
                print(log, "(nan)")
                
            elif torch.isinf(x).any():
                print(log, "(inf)")
                
