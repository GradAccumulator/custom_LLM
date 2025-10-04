import torch
from src.modules import (
    Linear, 
    Dropout, 
    LayerNorm, 
    SelfAttention, 
    Residual, 
    Scale, 
    To, 
    Layers,
    Embedding,
    PositionalEmbedding,
    Module,
    Sequential
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.utils.types import ActivationFunction
    from src.utils.types import koTokenizer

class MyTransformer(Module):
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
            qk_std_init:float, 
            scale_res_atten:float, 
            scale_res_ffn:float, 
            device:str, 
            dtype:torch.dtype, 
            tokenizer:"koTokenizer", 
            forward_dtype=torch.float16
        ):
        super().__init__(device, dtype, seq_len=seq_len, num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, act_fn=act_fn, p_dropout=p_dropout, LN_eps=LN_eps, qk_std_init=qk_std_init, scale_res_atten=scale_res_atten, scale_res_ffn=scale_res_ffn, tokenizer=tokenizer, forward_dtype=forward_dtype, device=device, dtype=dtype)
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.act_fn = act_fn
        self.p_dropout = p_dropout
        self.LN_eps = LN_eps
        self.qk_std_init = qk_std_init
        self.scale_res_atten = scale_res_atten
        self.scale_res_ffn = scale_res_ffn
        self.tokenizer = tokenizer
        self.residual = Residual()
        
        self.model = Sequential(
            embedding=Embedding(self.tokenizer.vocab_size, self.d_model),
            to=To(forward_dtype) if forward_dtype != self.dtype else lambda x: x,
            pos_embedding=PositionalEmbedding(self.seq_len, self.d_model),
            dropout=Dropout(p=self.p_dropout),
            transformer_layers=Layers(
                res1_save=self.residual.save(),
                LN1=LayerNorm(self.d_model, eps=self.LN_eps),
                attention=SelfAttention(self.d_model, self.num_heads, seq_len=self.seq_len, qk_std_init=None, learnable_qk_std=True, dropout_p=self.p_dropout),
                dropout1=Dropout(p=self.p_dropout),
                #scale_res_atten=Scale(self.scale_res_atten, learnable=True),
                res1_sum=self.residual.sum(),
                
                res2_save=self.residual.save(),
                LN2=LayerNorm(self.d_model, eps=self.LN_eps),
                FFN_in=Linear(self.d_model, self.d_ffn),
                act_fn=self.act_fn,
                FFN_out=Linear(self.d_ffn, self.d_model),
                dropout2=Dropout(p=self.p_dropout),
                #scale_res_ffn=Scale(self.scale_res_ffn, learnable=True),
                res2_sum=self.residual.sum(),
            num_layers=self.num_layers, save_inputs=False)
            #weight tying
        )
        self.model.to(device=self.device, dtype=self.dtype)
    
    def forward(self, x):
        model_out = self.model(x)
        logits = model_out@self.model['embedding'].table.T.to(model_out.dtype)
        return logits
        
    def decode_pred(self, pred:torch.Tensor):
        '''pred는 softmax거친 확률값'''
        return self.tokenizer.decode(pred.argmax(dim=-1).tolist())
    def decode(self, x:torch.Tensor):
        return self.tokenizer.decode(x)
    
    def encode(self, x:list[str],*, is_dataset=False, max_length=10000) -> torch.Tensor:
        if not is_dataset:
            return torch.from_numpy(self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=False, max_length=max_length)).to(self.device)
        input,label = self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=True, max_length=max_length)
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
        return (input, label)
    def encode_as_list(self, x:list[str], is_dataset=False) -> list[int]:
        return self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=is_dataset)