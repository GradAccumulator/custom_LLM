from assets.configs.default_settings import transformer_config
from assets.configs.trainer_config import TrainerConfig

from src.training.trainer import Trainer
from src.models.transformer import Transformer
from src.training.Optimizers import Adam, SGD
from src.nlp.NLP_preprocessor import NLP_Preprocessor

if __name__ == '__main__':
    model = Transformer(transformer_config)
    
    max_lr = 8e-5
    optimizer = Adam(model.parameters(), max_lr, 0.9, 0.95, 1e-2, eps=1e-8, clip_grad=5)
    
    preprocessor = NLP_Preprocessor(model, )
    
    TrainerConfig()