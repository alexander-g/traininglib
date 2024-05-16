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



