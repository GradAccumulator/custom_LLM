import torch
import msvcrt
from src.utils.dev_utils import timer, print_log
from src.utils.processing.data_processing import clear_memory
import bdb
import torchmetrics
from typing import Literal
import math
from colorama import Fore, Style, init as colorama_init
import datetime
colorama_init()

def alert(msg):
    from pygame import mixer
    from win10toast import ToastNotifier
    from time import sleep
    import threading
    toaster = ToastNotifier()
    print(msg)
    mixer.init()
    mixer.music.load("assets/bigdata/999C40345D542B9E23.mp3")
    toaster.show_toast("Warning!!!!!", msg, duration=5)
    for _ in range(16):
        mixer.music.play()
        if msvcrt.kbhit() and msvcrt.getch() == b'q':
            break
        sleep(16)


class Trainer:
    STOP = "STOP"
    EXIT = "EXIT"
    def __init__(self, model, optimizer, train_loader, val_loader, logger, config, loss_fn, tboard_manager=None, scheduler=None,*, validation=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.current_epoch = 0
        self.logger = logger
        self.tboard_manager = tboard_manager
        self.config = config
        self.max_epoch = self.config.max_epoch
        self.loss_fn = loss_fn
        self.grad_accumulation = self.config.grad_accumulation
        self.current_step = 0
        self.validation = validation
        self.acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=self.model.tokenizer.vocab_size).to(self.model.device)
        self.effective_steps_per_epoch = math.ceil(self.config.steps_per_epoch/self.grad_accumulation)
        self.nstep = 27
        self.nloss = 21
        self.nacc = 28
        self.ngnorm = 21
        self.effective_step = 0
        self.validation_interval = self.config.validation_interval*self.grad_accumulation
        self.logger.write(
            f'<<{datetime.datetime.now().isoformat()[:-7]}>>\n'
            f'grad_accummulation:{self.grad_accumulation}\n'
            f'warmup_rate:{self.config.warmup_steps_rate*100:.1f}%\n'
            f'warmup_steps(effective):{self.config.warmup_steps}\n'
            f'total_steps(effective):{self.config.total_steps}\n'
            f'effective_steps_per_epoch:{self.effective_steps_per_epoch}\n'
            f'max_grad_norm:{self.optimizer.max_grad_norm}\n'
            f'min_lr&max_lr:{self.scheduler.min_lr:.3g} to {self.scheduler.max_lr:.3g}\n'
            f'adamW (wd,eps,beta1,beta2):({self.optimizer.weight_decay},{self.optimizer.eps},{self.optimizer.b1},{self.optimizer.b2})\n'
        )

    def validate(self) -> tuple[float, float]:
        avg_loss = 0.0
        num_total_correct = 0
        num_total_tokens = 0
        clear_memory()
        self.model.eval()
        with torch.no_grad():
            for (val_x, val_label) in self.val_loader:
                    logits = self.model(val_x)
                    size_rate = self.val_loader.current_size_rate()
                    avg_loss += self.loss_fn(logits,val_label).item()*size_rate
                    num_total_correct += (logits.argmax(dim=-1)==val_label).float().sum().item()
                    num_total_tokens += val_label.numel()
        avg_acc = num_total_correct/num_total_tokens
        clear_memory()
        self.model.train()
        log = f'effective step:{self.effective_step}/{self.effective_steps_per_epoch}'.ljust(self.nstep)+ \
            f'| val   loss:{avg_loss:.4g}'.ljust(self.nloss)+ \
            f'| val   accuracy:{avg_acc*100:.3g}%'.ljust(self.nacc)
        print(Fore.GREEN+log+'| validated!!'+Style.RESET_ALL)
        self.logger.write(datetime.datetime.now().isoformat()+" | "+log+
            "\n\t| validated!!"
        )
        return (avg_loss, avg_acc*100)
    
    def debug(self, logits, train_x, train_label, loss, val_loss, val_acc, *args, **kwargs) -> tuple[float, float]|Literal["STOP"]|Literal["EXIT"]:
        if msvcrt.kbhit():
            key = msvcrt.getch().lower()
            #quit training(not saves checkpoint)
            if key == b'q':      
                print("stop training")
                return self.STOP
            #exit training(saves checkpoint)
            elif key == b'e':    
                print("exit training")
                return self.EXIT
            #debug
            elif key == b'd':    
                print("start debugging")
                breakpoint()
            #print prediction
            elif key == b'a':
                print(
                    "====================================================== model prediction ======================================================\n"
                    f"{self.model.decode_pred(logits)}\n"
                    "==============================================================================================================================\n"
                )
            #validate
            elif key == b'v':    
                print("start validation")
                return self.validate()
            #save
            elif key == b's':    
                self.save_checkpoint()
        return (val_loss, val_acc)

    @property
    def grad_accumulation(self):
        return self._step_interval
    @grad_accumulation.setter
    def grad_accumulation(self, value):
        self._step_interval = value
        if self._step_interval>1:
            self.step = self.step_with_term
        else:
            self.step = self.step_with_no_term

    def tboard_update_loss(self, loss, val_loss, logits, label, val_acc, total_acc_in_accum):
        if self.tboard_manager and (self.current_step%self.config.tboard_update_interval==0 or self.current_step==1):
            with torch.no_grad():
                avg_acc = (total_acc_in_accum/self.grad_accumulation).item()*100
                self.tboard_manager.add_scalars(
                    ('loss/train'       , loss              , self.effective_step, True),
                    ('loss/validation'  , val_loss          , self.effective_step, True),
                    ('lr'               , self.optimizer.lr , self.effective_step, False),
                    ("acc/train"        , avg_acc           , self.effective_step, True),
                    ("acc/validation"   , val_acc           , self.effective_step, True)
                )
                        
    def check_nan_and_inf(self, tensor, name):
        if tensor.isnan().any():
            alert(f"[Trainer] {name} is nan!!!!!")
            raise ValueError(f"[Trainer] {name} is nan!!!!!")
        
        elif tensor.isinf().any():
            alert(f"[Trainer] {name} is inf!!!!!")
            raise ValueError(f"[Trainer] {name} is inf!!!!!")

    def step_and_logging(self):
        self.effective_step += 1
        self.weight_norm = (sum((p**2).sum() for p in self.model.parameters())).sqrt()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()
        if self.effective_step%self.config.log_interval==0:
            log = (
                f'effective step:{self.effective_step}/{self.effective_steps_per_epoch}'.ljust(self.nstep)+
                f'| train loss:{self.avg_loss_in_accum:.4g}'.ljust(self.nloss)+
                f'| train accuracy:{(self.total_acc_in_accum/self.grad_accumulation).item()*100:.3g}%'.ljust(self.nacc)+' '
            )
            log_norms_and_lr = (
                f'| grad_norm:{self.optimizer.prev_grad_norm.item():.4g}'.ljust(self.ngnorm)+
                f'| weight_norm:{self.weight_norm.item():.4g}'.ljust(self.ngnorm-2)+
                f'| lr:{self.scheduler.prev_lr:.5g}'
            )
            print(log+log_norms_and_lr)
            self.logger.write(datetime.datetime.now().isoformat()+" | "+log+'\n\t'+log_norms_and_lr)
        self.avg_loss_in_accum = 0
        self.total_acc_in_accum = 0
    
    def step_with_term(self):
        if (self.current_step%self.grad_accumulation)==0 or self.train_loader.is_last_batch():
            self.step_and_logging()
    def step_with_no_term(self):
        self.step_and_logging()
        
    def forward_and_compute_loss(self, train_x, train_label) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(train_x)
        loss = self.loss_fn(logits, train_label)
        self.check_nan_and_inf(loss, "loss")
        return logits, loss
        
    def train_one_epoch(self):
        self.model.train()
        val_loss = torch.nan
        val_acc = torch.nan
        self.avg_loss_in_accum = 0
        self.total_acc_in_accum = 0
        debug_fn = lambda:None
        for (train_x, train_label) in self.train_loader:
                self.current_step+=1
                
                if self.validation and (self.current_step%self.validation_interval==0 or self.current_step==1):
                    val_loss, val_acc = self.validate()
                
                logits, loss = self.forward_and_compute_loss(train_x, train_label)
                loss = loss/self.grad_accumulation

                #backprop
                loss.backward()        
                
                #record loss and acc
                self.avg_loss_in_accum += loss.item()
                self.total_acc_in_accum += self.acc_fn(logits.view(-1, logits.size(-1)), train_label.view(-1))
                
                #tboard log
                self.tboard_update_loss(self.avg_loss_in_accum, val_loss, logits, train_label, val_acc, self.total_acc_in_accum)
                
                #step
                self.step()

                #debug
                debug_fn()
                debug_out = self.debug(logits, train_x, train_label, loss, val_loss, val_acc)
                if debug_out in [self.STOP, self.EXIT]:
                    return debug_out
                val_loss, val_acc = debug_out
        return False
    def train(self):
        try:
            print(f"최대 에폭:{self.max_epoch}")
            print(f"steps_per_epoch:{self.config.steps_per_epoch}")
            print(f"warmup steps: {self.config.warmup_steps_real}")
            print(f"total steps: {self.config.total_steps_real}")
            print_log(
                "======================================================"
                            "   start training   "
                "======================================================"
            )
            for _ in range(self.current_epoch, self.max_epoch):
                if debug:=self.train_one_epoch() != False:
                    self.quit_training(debug==self.STOP)
                    return
                print("======================================================   중간 평가   ======================================================")
                print(f"{{{self.current_epoch}}} validation loss : {self.validate():.10g}")
                print("==============================================================================================================================")
                self.current_epoch+=1
                self.current_step = 0
                self.effective_step = 0
                self.save_checkpoint()
                self.train_loader.shuffle()
        except (KeyboardInterrupt, SystemExit, bdb.BdbQuit,):
            raise
        except Exception as e:
            breakpoint()
            raise e

    def quit_training(self, save=True):
        self.tboard_manager.close()
        if save:
            self.save_checkpoint()
    
    def Error(self):
        self.tboard_manager.close()

    def state_dict(self):
        return {
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'train_loader':self.train_loader.state_dict(),
            'val_loader':self.val_loader.state_dict(),
            'loss_fn':self.loss_fn.state_dict(),
            'current_epoch':self.current_epoch,
            'current_step':self.current_step,
            'validation':self.validation,
            'avg_loss_in_accum':self.avg_loss_in_accum,
            'total_acc_in_accum':self.total_acc_in_accum,
        }

    def save_checkpoint(self):
        print("checkpoint 저장 시작")
        state = self.state_dict()
        torch.save(state, self.config.TRAINER_SAVE_PATH)
        if self.logger:
            self.logger.save()
        print("checkpoint 저장 완료")
    @staticmethod
    def load_checkpoint(config, logger, loss_fn_type,  optimizer_type, model_type, dataloader_type, act_fn,scheduler_type=None, warmup_type=None, decay_type=None, tboard_manager=None):
        state = torch.load(config.TRAINER_SAVE_PATH)
        model = model_type.load_state(config.tokenizer,act_fn,state['model'])
        optimizer=optimizer_type.load_state(model.params,state['optimizer'])
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler_type.load_state(optimizer, warmup_type, decay_type, state['scheduler']) if scheduler_type else None,
            train_loader=dataloader_type.load_state(state['train_loader']),
            val_loader=dataloader_type.load_state(state['val_loader']),
            logger=logger,
            loss_fn=loss_fn_type.load_state(state['loss_fn']),
            tboard_manager=tboard_manager,
            validation=state['validation'],
            config=config
        )
        trainer.current_epoch = state['current_epoch']
        trainer.current_step = state['current_step']
        trainer.logger.load_prev_logs()
        trainer.avg_loss_in_accum = state['avg_loss_in_accum']
        trainer.total_acc_in_accum = state['total_acc_in_accum']
        trainer.effective_step = optimizer.step
        return trainer
    
    
    def __del__(self):
        self.tboard_manager.close()
        print(f"last epoch: {self.current_epoch}")
        print(f"last step: {self.current_step}")
        print(f"last lr: {self.optimizer.lr}")
        print(f"last train loss: {self.logger.moving_average('train', 1)}")
        print(f"last validation loss: {self.logger.moving_average('validation', 1)}")