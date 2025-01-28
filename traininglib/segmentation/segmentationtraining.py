import typing as tp
import argparse
import numpy as np
import torch, torchvision

from ..trainingtask import Loss, Metrics
from .. import datalib, modellib
from ..datalib import Color, convert_rgb_to_mask
from .patchwisemodel import PatchedCachingDataset, PatchwiseTrainingTask, TensorPair




class SegmentationTask(PatchwiseTrainingTask):
    def __init__(
        self, 
        *a,
        colors:         tp.List[Color],
        ignore_colors:  tp.List[Color]|None         = None,
        pos_weight:     float = 1.0,
        margin_weight:  float = 0.0,
        **kw
    ):
        assert len(colors) == 1, NotImplementedError('TODO: implement multiclass training')
        super().__init__(*a, **kw)
        self.colors        = colors
        self.ignore_colors = ignore_colors
        self.pos_weight    = torch.as_tensor(pos_weight, dtype=torch.float32)
        self.margin_weight = margin_weight

    def forward_step(self, batch:TensorPair, augment:bool) -> tp.Tuple[Loss, Metrics]:
        '''Code re-use for training and validation'''
        batch = self.prepare_batch(batch, augment)
        #batch = datalib.to_device(*batch, device=self.device) # type: ignore [assignment]
        x,t = batch

        #if augment:
        #    x,t = self.augment((x,t))
        
        t_rgb = t
        t = convert_rgb_to_mask(t_rgb, self.colors)
        weight = None
        if self.ignore_colors is not None:
            weight = 1 - convert_rgb_to_mask(t_rgb, self.ignore_colors)
        
        y    = self.basemodule(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y, t, weight, pos_weight=self.pos_weight.to(y.device)
        )
        logs = {'bce': float(loss)}
        if self.margin_weight > 0:
            margin_loss = margin_loss_fn(y, (t==1)) * self.margin_weight
            loss = loss + margin_loss
            logs['margin'] = float(margin_loss)
        
        #TODO: per class
        # accuracy_map = ( t == (y > 0.5) ).float()
        # accuracies   = accuracy_map.reshape(len(y), -1).mean(-1)
        # logs['acc']  = float(accuracies.mean())

        return loss, logs


    
    

# TODO: this is not batch-aware!
def margin_loss_fn(y:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    '''Loss pushing positive and negative samples apart'''
    assert t.ndim == 4 and y.ndim == 4
    assert y.shape[1] == 1, 'Multiclass outputs not supported'
    assert t.shape == y.shape
    assert t.dtype == torch.bool

    y     = y.sigmoid()
    y_pos = y[t]
    y_neg = y[~t]
    n     = min(len(y_pos), len(y_neg))
    y_pos = y_pos[torch.randperm(len(y_pos))[:n]]
    y_neg = y_neg[torch.randperm(len(y_neg))[:n]]

    return torch.nn.functional.margin_ranking_loss(
        y_pos, y_neg, torch.ones_like(y_pos), margin=1.0
    )
    
@torch.jit.script_if_tracing
def margin_loss_multilabel(
    y:torch.Tensor, 
    t:torch.Tensor, 
    logits:bool,
    margin:float = 1.0
) -> torch.Tensor:
    '''Loss pushing positive and negative samples apart,
       applied individually per batch and per channel'''
    assert t.ndim == y.ndim == 4
    assert t.shape == y.shape
    assert t.dtype == torch.bool

    if logits:
        y = y.sigmoid()

    all_y_pos:tp.List[torch.Tensor] = []    
    all_y_neg:tp.List[torch.Tensor] = []
    for b in range(t.shape[0]):
        for c in range(t.shape[1]):
            y_bc = y[b,c]
            t_bc = t[b,c]
            y_pos = y_bc[t_bc]
            y_neg = y_bc[~t_bc]
            n     = min(len(y_pos), len(y_neg))
            y_pos = y_pos[torch.randperm(len(y_pos))[:n]]
            y_neg = y_neg[torch.randperm(len(y_neg))[:n]]

            all_y_pos.append(y_pos)
            all_y_neg.append(y_neg)

    all_y_pos = torch.cat(all_y_pos)
    all_y_neg = torch.cat(all_y_neg)
    loss = torch.nn.functional.margin_ranking_loss(
        all_y_pos,
        all_y_neg,
        torch.ones_like(all_y_pos),
        margin,
        reduction = 'none',
    )
    return torch.nan_to_num( torch.nanmean(loss) )





def start_segmentation_training_from_cli_args(
    args:       argparse.Namespace,
    model:      'SegmentationModel',  # type: ignore
    task_kw:    tp.Dict[str, tp.Any]  = {},
    fit_kw:     tp.Dict[str, tp.Any]  = {},
) -> bool:
    '''`SegmentationModel.start_training()` with basic config provided by
       command line arguments from `args.base_segmentation_training_argparser()`'''
    
    trainsplit = datalib.load_file_pairs(args.trainsplit)
    
    valsplit = None
    if args.valsplit is not None:
        valsplit = datalib.load_file_pairs(args.valsplit)
    
    task_kw = {
        'pos_weight':    args.pos_weight, 
        'margin_weight': args.margin_weight,
        'rotate':        args.rotate,
    } | task_kw
    return modellib.start_training_from_cli_args(
        args, 
        model, 
        trainsplit, 
        valsplit, 
        task_kw=task_kw
    )


def precision_recall(
    ypred: torch.Tensor, 
    ytrue: torch.Tensor,
    n_classes: int,
) -> tp.Tuple[tp.List[float], tp.List[float]]:
    assert ypred.dtype == ytrue.dtype == torch.int64
    assert ypred.shape == ytrue.shape

    precision: tp.List[float] = []
    recall:    tp.List[float] = []

    # binary if n_classes == 1
    for cls in range(n_classes) if n_classes > 1 else [1]:
        preds_cls   = (ypred == cls)
        targets_cls = (ytrue == cls)

        TP = int( (preds_cls & targets_cls).sum() )
        FP = int( (preds_cls & ~targets_cls).sum() )
        FN = int( (~preds_cls & targets_cls).sum() )

        p = TP / (TP + FP) if (TP + FP) > 0 else 0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0

        precision.append(p)
        recall.append(r)

    return precision, recall
