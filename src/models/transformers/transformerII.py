import torch
from ...modules.sequential import Sequential, Layers
from ...modules import Linear, Dropout, LayerNorm, SelfAttention, Residual, Scale, To
from ...nlp.embedding import Embedding

class TransformerII:
    def __init__(self, seq_len, num_layers, d_model, num_heads, d_ffn, act_fn, p_dropout, LN_eps, scale_qk, scale_res_atten, scale_res_ffn, device, dtype, tokenizer, forward_type=torch.float16):
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.act_fn = act_fn
        self.p_dropout = p_dropout
        self.LN_eps = LN_eps
        self.scale_qk = scale_qk
        self.scale_res_atten = scale_res_atten
        self.scale_res_ffn = scale_res_ffn
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.residual = Residual()
        self.embedding = Embedding(self.tokenizer.vocab_size, self.d_model)
        
        self.model = Sequential(
            self.embedding,
            Embedding(self.seq_len, self.d_model),
            To(forward_type),
            Dropout(self.p_dropout),
            Layers(
                self.residual.save(),
                LayerNorm(self.d_model, self.LN_eps),
                SelfAttention(self.num_heads, self.d_model, self.d_ffn, self.act_fn, self.scale_qk, self.scale_res_atten),
                Dropout(self.p_dropout),
                Scale(self.scale_res_atten, learnable=True),
                self.residual.sum(),
                
                self.residual.save(),
                LayerNorm(self.d_model, self.LN_eps),
                Linear(self.d_model, self.d_ffn),
                Dropout(self.p_dropout),
                Scale(self.scale_res_ffn, learnable=True),
                self.residual.sum(),
            num_layers=self.num_layers)
            #weight tying
        )
        self.model.to(device=self.device, dtype=self.dtype)
        
    def forward(self, x):
        model_out = self.model(x)
        logits = model_out@self.embedding.weight.T
        return logits
    
    def __call__(self, x):
        return self.forward(x)
    
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'embedding': self.embedding.state_dict(),
        }
    
    def load_state(self, state:dict):
        self.model.load_state_dict(state['model'])
        self.embedding.load_state_dict(state['embedding'])
    
    def save(self, path):
        torch.save(self.state_dict(), path)