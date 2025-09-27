import torch
from src.modules.sequential import Sequential
from src.modules import Linear, Dropout, LayerNorm, SelfAttention, Residual, Scale, To, Layers
from src.nlp import Embedding, PositionalEmbedding
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
            qk_std_init:float, 
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
        self.qk_std_init = qk_std_init
        self.scale_res_atten = scale_res_atten
        self.scale_res_ffn = scale_res_ffn
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.residual = Residual()
        
        self.model = Sequential(
            embedding=Embedding(self.tokenizer.vocab_size, self.d_model),
            to=To(forward_type),
            pos_embedding=PositionalEmbedding(self.seq_len, self.d_model),
            dropout=Dropout(p=self.p_dropout),
            transformer_layers=Layers(
                res1_save=self.residual.save(),
                LN1=LayerNorm(self.d_model, eps=self.LN_eps),
                to1=To(torch.float32),
                attention=SelfAttention(self.d_model, self.num_heads, seq_len=self.seq_len, qk_std_init=qk_std_init, learnable_qk_std=True, dropout_p=self.p_dropout),
                to2=To(forward_type),
                dropout1=Dropout(p=self.p_dropout),
                #scale_res_atten=Scale(self.scale_res_atten, learnable=True),
                res1_sum=self.residual.sum(),
                
                res2_save=self.residual.save(),
                LN2=LayerNorm(self.d_model, eps=self.LN_eps),
                FFN_in=Linear(self.d_model, self.d_ffn),
                act_fn=self.act_fn,
                FFN_out=Linear(self.d_ffn, self.d_model),
                dropout=Dropout(p=self.p_dropout),
                #scale_res_ffn=Scale(self.scale_res_ffn, learnable=True),
                res2_sum=self.residual.sum(),
            num_layers=self.num_layers, save_inputs=False)
            #weight tying
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    
    def forward(self, x):
        model_out = self.model(x)
        logits = model_out@self.model['embedding'].table.T.to(model_out.dtype)
        return logits
    
    def __call__(self, x):
        return self.forward(x)
    
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
        }
    
    def load_state(self, state:dict):
        self.model.load_state_dict(state['model'])
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def decode_pred(self, x:torch.Tensor):
        '''x는 softmax거친 확률값'''
        return self.tokenizer.decode(x.argmax(dim=-1).tolist())
    def decode(self, x:torch.Tensor):
        return self.tokenizer.decode(x)
    
    def parameters(self):
        return self.model.parameters()
    def parameters_as_dict(self):
        return self.model.named_parameters() 
    
    def encode(self, x:list[str],*, is_dataset=False, max_length=10000) -> torch.Tensor:
        if not is_dataset:
            return torch.from_numpy(self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=False, max_length=max_length)).to(self.device)
        input,label = self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=True, max_length=max_length)
        input = torch.from_numpy(input)
        label = torch.from_numpy(label)
        return (input, label)
    def encode_as_list(self, x:list[str], is_dataset=False) -> list[int]:
        return self.tokenizer.encode(x, seq_len=self.seq_len, is_dataset=is_dataset)