import os
import typing as tp
import warnings

import numpy as np
import torch, torchvision


Loss    = torch.Tensor
Metrics = tp.Dict[str, float]

class TrainingTask(torch.nn.Module):
    """Base class for training"""

    def __init__(self, basemodule:torch.nn.Module):
        super().__init__()
        self.basemodule = basemodule

    def training_step(self, batch) -> tp.Tuple[Loss, Metrics]:
        """Abstract training step function. Should return a loss scalar and a dictionary with metrics"""
        raise NotImplementedError()

    def validation_step(self, batch) -> Metrics:
        """Abstract validation step function. Should return a dictionary with metrics"""
        raise NotImplementedError()
    
    def create_dataloaders(
        self, 
        trainsplit: tp.List[tp.Any], 
        valsplit:   tp.List[tp.Any]|None = None, 
        **ld_kw
    ) -> tp.Tuple[tp.Iterable, tp.Iterable|None]:
        '''Abstract function that creates training and optionally validation data loaders.'''
        raise NotImplementedError()

    def configure_optimizers(
        self, 
        epochs:        int, 
        lr:            float,
        warmup_epochs: int = 0, 
        optimizer:     tp.Literal['adamw']  = 'adamw', 
        scheduler:     tp.Literal['cosine'] = 'cosine'
    ) -> tp.Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Optimizer and scheduler configuration"""
        assert warmup_epochs >= 0
        assert optimizer == 'adamw', NotImplemented

        optim: torch.optim.Optimizer
        optim = torch.optim.AdamW(self.parameters(), lr=lr)
        schedulers:tp.List[torch.optim.lr_scheduler.LRScheduler] = []

        if warmup_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optim, start_factor=1/100, total_iters=warmup_epochs
                )
            )

        if scheduler == 'cosine':
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, epochs - warmup_epochs, eta_min=lr / 1000
                )
            )
        
        sched = None
        if len(schedulers) == 1:
            sched = schedulers[0]
        elif len(schedulers) > 1:
            sched = torch.optim.lr_scheduler.SequentialLR(
                optim, schedulers, milestones=[warmup_epochs]
            )
        return optim, sched # type: ignore

    @property
    def device(self) -> torch.device:
        """Convenience property to get the device of the task/model"""
        return next(self.parameters()).device

    def train_one_epoch(
        self, 
        loader:    tp.Iterable, 
        optimizer: torch.optim.Optimizer, 
        scaler:    torch.cuda.amp.GradScaler, 
        epoch:     int, 
        scheduler: torch.optim.lr_scheduler.LRScheduler|None = None, 
        callback   = None,
        amp:       bool = False,
    ) -> None:
        n_batches = len(loader)
        loader_it = iter(loader)
        for i in range(n_batches):
            batch = next(loader_it)
            optimizer.zero_grad()

            with torch.autocast("cuda", enabled=amp):
                loss, logs = self.training_step(batch)
            logs["lr"] = optimizer.param_groups[0]["lr"]

            loss = scaler.scale(loss)
            loss.backward()
            scaler.step(optimizer)
            scaler.update()

            if callback is not None:
                callback.on_batch_end(logs, i, n_batches)  # type: ignore [arg-type]
        
        #TODO: scheduler.step inside loop
        if scheduler:
            scheduler.step()

    def eval_one_epoch(self, loader: tp.Iterable, callback=None, amp:bool=False):
        for i, batch in enumerate(loader):
            with torch.autocast("cuda", enabled=amp):
                logs = self.validation_step(batch)
            if callback is not None:
                callback.on_batch_end(logs, i, len(loader))  # type: ignore [arg-type]

    def fit(
        self,
        trainsplit:     tp.List[tp.Any],
        valsplit:       tp.List[tp.Any]|None = None,
        epochs:         int   = 30,
        warmup_epochs:  int   = 0,
        lr:             float = 1e-3,
        optimizer:      tp.Literal['adamw']  = 'adamw',
        scheduler:      tp.Literal['cosine'] = 'cosine',
        val_freq:       int      = 1,
        batch_size:     int      = 8,
        checkpoint_dir: str|None = None,
        callback:       tp.Callable|None  = None,
        device:         str      = "cuda" if torch.cuda.is_available() else "cpu",
        amp:            bool     = False,
    ):
        """
        Training entry point.
            trainsplit:     List of files to train on. Passed to self.create_dataloaders()
            valsplit:       Optional list of files for validation
            epochs:         Number of training epochs
            warmup_epochs:  Number of epochs to ramp up the learning rate from 0
            lr:             Base learning rate
            optimizer:      Training optimization algorithm
            scheduler:      Learning rate schedule
            val_freq:       How often to run validation (in epochs)
            checkpoint_dir: Optional path to where model, training state and code is stored
            device:         Device to run training on
            amp:            Automatic mixed precision
        """
        cb: TrainingProgressCallback | PrintMetricsCallback  # for mypy
        if callback is not None:
            cb = TrainingProgressCallback(callback, epochs)
        else:
            logfile = \
                None if checkpoint_dir is None \
                    else os.path.join(checkpoint_dir, 'log.txt')
            cb = PrintMetricsCallback(logfile)
        
        ld_kw = {'batch_size':batch_size}
        ld_train, ld_val = self.create_dataloaders(trainsplit, valsplit, **ld_kw)
        if ld_val is not None:
            #quick test to ensure validation_step() is actually implemented
            assert self.__class__.validation_step != TrainingTask.validation_step, \
                'Validation data provided but validation_step() is not implemented'

        optim, sched = self.configure_optimizers(
            epochs, lr, warmup_epochs, optimizer, scheduler
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        torch.cuda.empty_cache()

        best_loss = np.inf

        #TODO: backup code

        try:
            self.__class__.stop_requested = False

            for e in range(epochs):
                self.to(device)
                if self.__class__.stop_requested:
                    break
        
                self.train().requires_grad_(True)
                self.train_one_epoch(ld_train, optim, scaler, e, sched, cb, amp)

                self.eval().requires_grad_(False)
                if ld_val and (e % val_freq) == 0:
                    self.eval_one_epoch(ld_val, cb, amp)


                #TODO: save task as checkpoint to be able to resume training, not model
                #if checkpoint_dir is not None and e % checkpoint_freq == 0:
                #    self.basemodule.save(f"{checkpoint_dir}/checkpoint-{e:03d}.pt.zip")

                if checkpoint_dir is not None and "validation_loss" in cb.logs:
                    validation_loss = np.nanmean(cb.logs["validation_loss"])
                    if validation_loss <= best_loss:
                        self.basemodule.save(f"{checkpoint_dir}/best.pt.zip")
                        best_loss = validation_loss

                cb.on_epoch_end(e)
        finally:
            self.zero_grad(set_to_none=True)
            self.eval().cpu().requires_grad_(False)
            torch.cuda.empty_cache()
        
        #TODO: save final model
        return None

    stop_requested:bool = False

    # XXX: class method to avoid boilerplate code
    @classmethod
    def request_stop(cls):
        cls.stop_requested = True


class PrintMetricsCallback:
    """Prints metrics after each training epoch in a compact table"""

    def __init__(self, logfile:tp.Optional[str] = None):
        self.epoch = 0
        self.logs = {}
        self.logfile = logfile


    def on_epoch_end(self, epoch: int) -> None:
        self.epoch = epoch + 1
        self.print_accumulated_metrics(self.epoch, trim=False)
        self.log_metrics_to_file()
        self.logs = {}

    def on_batch_end(self, logs: tp.Dict, batch_i: int, n_batches: int) -> None:
        self.accumulate_logs(logs)
        percent = (batch_i + 1) / n_batches
        self.print_accumulated_metrics(percent, end='\r', trim=True)
    
    def format_metrics_string(self, progress:int|float) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            metrics_str = " | ".join(
                [f"{k}:{float(np.nanmean(v)):>8.5f}" for k, v in self.logs.items()]
            )
        progress_str = \
            f'{progress:.2f}' if type(progress) == float else f'{progress:4d}'
        print_str = f'[{progress_str}] {metrics_str}'
        return print_str

    def print_accumulated_metrics(
        self, 
        progress: int|float, 
        end:      str|None = None,
        trim:     bool     = False,
    ) -> None:
        print_str = self.format_metrics_string(progress)
        if trim:
            n_columns = os.get_terminal_size().columns
            print_str = print_str[:n_columns]
        print(print_str, end=end)

    def accumulate_logs(self, newlogs: tp.Dict) -> None:
        for k, v in newlogs.items():
            self.logs[k] = self.logs.get(k, []) + [v]
    
    def log_metrics_to_file(self) -> None:
        if self.logfile is None:
            return
        logdir = os.path.dirname(self.logfile)
        if not os.path.exists(logdir):
            # no checkpointdir because of --debug
            return
        
        print_str = self.format_metrics_string(self.epoch)
        open(self.logfile, 'a').write(print_str + '\n')


class TrainingProgressCallback:
    """Passes training progress as percentage to a custom callback function"""

    def __init__(self, callback_fn, epochs):
        self.n_epochs = epochs
        self.epoch = 0
        self.callback_fn = callback_fn

    def on_batch_end(self, logs, batch_i, n_batches):
        percent = (batch_i + 1) / (n_batches * self.n_epochs)
        percent += self.epoch / self.n_epochs
        self.callback_fn(percent)

    def on_epoch_end(self, epoch):
        self.epoch = epoch + 1
