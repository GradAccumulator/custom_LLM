from assets.configs.default_settings import (
    transformer_config, 
    preprocessor_config, 
    scheduler_config, 
    logger_config,
    tboard_manager_config,
)
from assets.configs.trainer_config import TrainerConfig

from src.nlp.NLP_preprocessor      import NLP_Preprocessor
from src.utils                     import DataLoader
from src.models.transformers       import (Transformer, TransformerII)
from src.training.Optimizers       import Adam
from src.training.Scheduler        import (Scheduler, warmups, decays)
from src.utils.logging             import (TBoardManager, Logger)
from src.training.trainer          import Trainer

from src.modules.loss_functions    import CrossEntropyLoss
from src.utils.dev_utils           import timer, parent_timer
from PATHES import PATHES

if __name__ == '__main__':
    with parent_timer("학습 전 설정"):
        with timer("모델 생성"):
            model = TransformerII(**transformer_config)
        
        with timer("데이터 전처리"):
            preprocessor = NLP_Preprocessor(model, preprocessor_config)
            train_x,train_label, val_x, val_label = preprocessor()
        
        with timer("데이터로더 생성"):
            train_loader = DataLoader(train_x, train_label, device='cuda')
            val_loader = DataLoader(val_x, val_label, device='cuda')
        
        with timer("optimizer & scheduler 생성"):
            max_lr = 1.5e-5
            optimizer = Adam(model.parameters_as_dict(), max_lr, 0.9, 0.95, 1e-3, eps=1e-8, clip_grad=1)
            
            scheduler = Scheduler(optimizer, warmups.LinearWarmup, scheduler_config.effective_warmup_steps, decays.CosineDecay, scheduler_config.effective_total_steps, min_lr = 8.5e-6)
        
        with timer("logger & tboard_manager 생성"):
            logger = Logger(logger_config.logs_dir)
            tboard_manager = TBoardManager(tboard_manager_config.logs_dir, window_size=20)
        
        with timer("loss 함수& trainer 생성"):
            loss_fn = CrossEntropyLoss(
                logit_input=True, 
                ignore_label_indices=[0], 
                stability_scale=0.1,
                weight_device='cuda',
            )
            
            trainer_config = TrainerConfig(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                logger=logger,
                loss_fn=loss_fn,
                warmup_steps_rate=scheduler_config.warmup_steps_rate,
                steps_per_epoch=scheduler_config.steps_per_epoch,
                total_epochs=scheduler_config.total_epochs,
                TRAINER_SAVE_PATH=PATHES.join(PATHES.ASSETS_MODELS, 'trainer.pth'),
                grad_accumulation=scheduler_config.grad_accumulation,
                tboard_manager=tboard_manager,
                scheduler=scheduler,
                validation=True,
                validation_interval=27,
                tboard_update_interval=1,
                log_interval=1
            )
            
            trainer = Trainer(config=trainer_config, **trainer_config)
    trainer.train()