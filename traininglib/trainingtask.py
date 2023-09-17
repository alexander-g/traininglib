import typing as tp
import numpy as np
import torch, torchvision


class TrainingTask(torch.nn.Module):
    """Base class for training"""

    def __init__(
        self,
        basemodule:     torch.nn.Module,
        lr:             float               = 1e-3,
        amp:            bool                = False,
        val_freq:       int                 = 1,
        callback:       tp.Callable | None  = None,
        optimizer:      tp.Literal['adamw'] = 'adamw',
        lr_scheduler:   tp.Literal['cosine']= 'cosine',
    ):
        super().__init__()
        self.basemodule         = basemodule
        self.lr                 = lr
        self.progress_callback  = callback
        self.val_freq           = val_freq
        self.optimizer          = optimizer
        self.lr_scheduler       = lr_scheduler
        # mixed precision
        self.amp = amp if torch.cuda.is_available() else False


    def training_step(self, batch) -> tp.Tuple[torch.Tensor, tp.Dict]:
        """Abstract training step function. Should return a loss scalar and a dictionary with metrics"""
        raise NotImplementedError()

    def validation_step(self, batch) -> tp.Dict:
        """Abstract validation step function. Should return a dictionary with metrics"""
        raise NotImplementedError()

    def configure_optimizers(self, epochs:int, warmup_epochs:int = 0):
        """Optimizer and scheduler configuration"""
        assert warmup_epochs >= 0

        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        schedulers = []

        if warmup_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optim, start_factor=1/100, total_iters=warmup_epochs
                )
            )

        if self.lr_scheduler == 'cosine':
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, epochs - warmup_epochs, eta_min=self.lr / 1000
                )
            )
        
        sched = None
        if len(schedulers) == 1:
            sched = schedulers[0]
        elif len(schedulers) > 1:
            sched = torch.optim.lr_scheduler.SequentialLR(
                optim, schedulers, milestones=[warmup_epochs]
            )
        return optim, sched

    @property
    def device(self) -> torch.device:
        """Convenience property to get the device of the task/model"""
        return next(self.parameters()).device

    def train_one_epoch(
        self, loader: tp.Iterable, optimizer, scaler, epoch, scheduler=None, callback=None
    ) -> None:
        
        # warmup for first epoch
        for i, batch in enumerate(loader):
            optimizer.zero_grad()

            with torch.autocast("cuda", enabled=self.amp):
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

    def eval_one_epoch(self, loader: tp.Iterable, callback=None):
        for i, batch in enumerate(loader):
            with torch.autocast("cuda", enabled=self.amp):
                logs = self.validation_step(batch)
            if callback is not None:
                callback.on_batch_end(logs, i, len(loader))  # type: ignore [arg-type]

    def fit(
        self,
        loader_train:   tp.Iterable,
        loader_valid:   tp.Iterable|None        = None,
        epochs:         int                     = 30,
        warmup_epochs:  int                     = 0,
        checkpoint_dir: str|None                = None,
        checkpoint_freq:int                     = 1,
        device:         str = "cuda" if torch.cuda.is_available() else "cpu",
        
        #__reraise:      bool = False,
    ):
        """
        Training entry point.
            loader_train: Iterable returning training batches, ideally torch.DataLoader
            loader_valid: Optional iterable returning validation batches
            epochs:       Number of training epochs
            checkpoint_dir: Optional path to where the model is saved after every epoch
            device:       device to run training on
        """
        callback: TrainingProgressCallback | PrintMetricsCallback  # for mypy
        if self.progress_callback is not None:
            callback = TrainingProgressCallback(self.progress_callback, epochs)
        else:
            callback = PrintMetricsCallback()

        optim, sched = self.configure_optimizers(epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        torch.cuda.empty_cache()

        best_loss = np.inf

        try:
            self.__class__.stop_requested = False

            for e in range(epochs):
                self.to(device)
                if self.__class__.stop_requested:
                    break
        
                self.train().requires_grad_(True)
                self.train_one_epoch(loader_train, optim, scaler, e, sched, callback)

                self.eval().requires_grad_(False)
                if loader_valid and (e % self.val_freq) == 0:
                    self.eval_one_epoch(loader_valid, callback)


                if checkpoint_dir is not None and e % checkpoint_freq == 0:
                    self.basemodule.save(f"{checkpoint_dir}/checkpoint-{e:03d}.pt.zip")

                if checkpoint_dir is not None and "validation_loss" in callback.logs:
                    validation_loss = np.nanmean(callback.logs["validation_loss"])
                    if validation_loss <= best_loss:
                        self.basemodule.save(f"{checkpoint_dir}/best.pt.zip")
                        best_loss = validation_loss

                metrics = {k: np.nanmean(v) for k, v in callback.logs.items()}

                callback.on_epoch_end(e)

        #except (Exception, KeyboardInterrupt) as e:
        #    if __reraise:
        #        raise e
        #    return e
        finally:
            self.zero_grad(set_to_none=True)
            self.eval().cpu().requires_grad_(False)
            torch.cuda.empty_cache()
        return None

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
            [f"{k}:{float(np.nanmean(v)):>9.5f}" for k, v in self.logs.items()]
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
