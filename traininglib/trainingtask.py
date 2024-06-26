import typing as tp
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
        
        # warmup for first epoch
        for i, batch in enumerate(loader):
            optimizer.zero_grad()

            with torch.autocast("cuda", enabled=amp):
                loss, logs = self.training_step(batch)
            logs["lr"] = optimizer.param_groups[0]["lr"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if callback is not None:
                callback.on_batch_end(logs, i, len(loader))  # type: ignore [arg-type]
        
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
            cb = PrintMetricsCallback()
        
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

    def __init__(self):
        self.epoch = 0
        self.logs = {}

    def on_epoch_end(self, epoch: int) -> None:
        self.epoch = epoch + 1
        self.logs = {}
        print()  # newline

    def on_batch_end(self, logs: tp.Dict, batch_i: int, n_batches: int) -> None:
        self.accumulate_logs(logs)
        percent = (batch_i + 1) / n_batches
        metrics_str = " | ".join(
            [f"{k}:{float(np.nanmean(v)):>8.5f}" for k, v in self.logs.items()]
        )
        print(f"[{self.epoch:04d}|{percent:.2f}] {metrics_str}", end="\r")

    def accumulate_logs(self, newlogs: tp.Dict) -> None:
        for k, v in newlogs.items():
            self.logs[k] = self.logs.get(k, []) + [v]


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
