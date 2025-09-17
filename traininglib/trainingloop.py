import argparse
import os
import typing as tp

import numpy as np
import torch

from . import util, datalib, modellib
from .trainingtask import Callback, PrintMetricsCallback, TrainingProgressCallback


Loss    = torch.Tensor
Metrics = tp.Dict[str, float]
LossAndMetrics = tp.Tuple[Loss, Metrics]





def train_one_epoch(
    training_step: torch.nn.Module,
    loader:    tp.Sequence, 
    optimizer: torch.optim.Optimizer, 
    scheduler: tp.Optional[torch.optim.lr_scheduler.LRScheduler] = None, 
    callback:  tp.Optional[Callback] = None,
    scaler:    tp.Optional[torch.cuda.amp.GradScaler] = None, 
    amp:       bool = False,
) -> None:

    n_batches = len(loader)
    loader_it = iter(loader)

    for i in range(n_batches):
        optimizer.zero_grad()

        batch = next(loader_it)
        with torch.autocast("cuda", enabled=amp):
            loss, logs = training_step.train()(batch)
        logs["lr"] = optimizer.param_groups[0]["lr"]

        if scaler is not None:
            loss = scaler.scale(loss)
        
        loss.backward()
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        if callback is not None:
            callback.on_batch_end(logs, i, n_batches)
        
    if scheduler is not None:
        scheduler.step()


def train_one_step(
    training_step: torch.nn.Module,
    load_iter: tp.Iterator, 
    optimizer: torch.optim.Optimizer, 
    scheduler: tp.Optional[torch.optim.lr_scheduler.LRScheduler] = None, 
    callback:  tp.Optional[tp.Callable] = None,
    scaler:    tp.Optional[torch.cuda.amp.GradScaler] = None, 
    amp:       bool = False,
) -> None:
    optimizer.zero_grad()

    batch = next(load_iter)
    with torch.autocast("cuda", enabled=amp):
        loss, logs = training_step(batch)
    logs["lr"] = optimizer.param_groups[0]["lr"]

    if scaler is not None:
        loss = scaler.scale(loss)
    loss.backward()
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
        
    if scheduler is not None:
        scheduler.step()



def train(
    training_step:   torch.nn.Module,
    training_loader: tp.Sequence,
    epochs: tp.Optional[int],
    steps:  tp.Optional[int] = None,
    lr:     float = 1e-3,
    device: tp.Optional[torch.device] = None,
    checkpoint_dir: tp.Optional[str]  = None,
    progress_callback: tp.Optional[tp.Callable[[float], None]] = None,
):
    assert epochs is not None or steps is not None, 'epochs or steps required'
    assert epochs is None or steps is None, 'epochs or steps, not both'
    if epochs is None:
        steps  = tp.cast(int, steps)
        epochs = np.ceil(steps / len(training_loader))
        epochs = int(epochs)

    cb:Callback
    if progress_callback is not None:
        cb = TrainingProgressCallback(progress_callback, epochs)
    else:
        logfile = \
            None if checkpoint_dir is None \
                else os.path.join(checkpoint_dir, 'log.txt')
        cb = PrintMetricsCallback(logfile)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_step.to(device)
    
    optimizer, scheduler = \
        create_optimizer(training_step, epochs, lr=lr, optimizer='adamw')
    try:
        for e in range(epochs):
            train_one_epoch(training_step, training_loader, optimizer, scheduler, cb)
            cb.on_epoch_end(e)
    finally:
        training_step.zero_grad(set_to_none=True)
        training_step.eval().cpu().requires_grad_(False)
        torch.cuda.empty_cache()



def create_optimizer(
    module: torch.nn.Module, 
    epochs: int, 
    lr:     float,
    warmup_epochs: int = 0, 
    optimizer:     tp.Literal['adamw']  = 'adamw', 
    scheduler:     tp.Literal['cosine'] = 'cosine'
) -> tp.Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Optimizer and scheduler configuration"""
    assert warmup_epochs >= 0
    assert optimizer == 'adamw', NotImplemented

    optim: torch.optim.Optimizer
    optim = torch.optim.AdamW(module.parameters(), lr=lr)
    schedulers:tp.List[torch.optim.lr_scheduler.LRScheduler] = []

    if warmup_epochs > 0:
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                optim, 
                start_factor=1/100, 
                total_iters=warmup_epochs
            )
        )
    if scheduler == 'cosine':
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, 
                epochs - warmup_epochs, 
                eta_min=lr / 1000
            )
        )
    
    sched = None
    if len(schedulers) == 1:
        sched = schedulers[0]
    elif len(schedulers) > 1:
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, 
            schedulers, 
            milestones=[warmup_epochs]
        )
    return optim, sched # type: ignore


def start_training_from_cli_args(
    args:          argparse.Namespace,
    train_step:    modellib.SaveableModule,
    train_dataset: tp.Iterable,
    ld_kw:         tp.Dict[str, tp.Any] = {},
):
    train_step, paths = util.prepare_for_training(train_step, args) # type: ignore
    ld:tp.Sequence = datalib.create_dataloader( # type: ignore
        train_dataset, 
        batch_size = args.batchsize,
        shuffle    = True,
        **ld_kw
    )

    train(train_step, ld, args.epochs, lr=args.lr, checkpoint_dir=paths.checkpointdir)
    
    train_step.save(paths.modelpath)
    if paths.modelpath_tmp is not None:
        os.remove(paths.modelpath_tmp)
    return train_step, paths


