import torch
from src.modules.sequential import Sequential
from src.modules import Linear, Dropout, LayerNorm, SelfAttention, Residual, Scale, To, Layers
from src.nlp.embedding import Embedding
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.utils.types import ActivationFunction
    from src.utils.types import koTokenizer

class TransformerII:
    def __init__(
            self, 
            seq_len:int, 
            num_layers:int, 
            d_model:int, 
            num_heads:int, 
            d_ffn:int, 
            act_fn:"ActivationFunction", 
            p_dropout:float, 
            LN_eps:float, 
            scale_qk:float, 
            scale_res_atten:float, 
            scale_res_ffn:float, 
            device:str, 
            dtype:torch.dtype, 
            tokenizer:"koTokenizer", 
            forward_type=torch.float16
        ):
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
                LayerNorm(self.d_model, eps=self.LN_eps),
                SelfAttention(self.d_model, self.num_heads, seq_len=self.seq_len, scale_qk_value=self.scale_qk, learnable_scale_qk=True, dropout_p=self.p_dropout),
                Dropout(self.p_dropout),
                Scale(self.scale_res_atten, learnable=True),
                self.residual.sum(),
                
                self.residual.save(),
                LayerNorm(self.d_model, eps=self.LN_eps),
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
        
    def decode_pred(self, x:torch.Tensor):
        '''x는 softmax거친 확률값'''
        return self.tokenizer.decode(x.argmax(dim=-1).tolist())
    def decode(self, x:torch.Tensor):
        return self.tokenizer.decode(x)
    
    def encode(self, x:list[str],*, for_eval=False, is_dataset=False, max_length=10000) -> torch.Tensor:
        if not is_dataset:
            return torch.from_numpy(self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=False, max_length=max_length), device=self.device)
        input,label = self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=True, max_length=max_length)
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
        return (input, label)
    def encode_as_list(self, x:list[str], is_dataset=False) -> list[int]:
        return self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=is_dataset)