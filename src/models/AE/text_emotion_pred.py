import torch
from src.modules import (
    GELU,
    Linear,
    Sequential,
    SelfAttention,
)

class AE:
    def __init__(self, d_in, text_vocab_size, emotion_vocab_size, d_model, latent_ratio, num_heads,*, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.text_vocab_size = text_vocab_size
        self.emotion_vocab_size = emotion_vocab_size
        self.d_model = d_model
        self.latent_ratio = latent_ratio
        self.d_in = d_in
        self.num_heads = num_heads
        self.d_latent = d_model*latent_ratio

        self.embedding = Linear(self.d_in, self.d_model)
        
        self.encoder = Sequential(
            Linear(self.d_model, self.d_latent//2),
            GELU(),
            Linear(self.d_latent//2, self.d_latent),
        )
        self.atten = SelfAttention(self.d_latent, self.num_heads, device=self.device, dtype=self.dtype)
        