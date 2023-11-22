import typing as tp
import numpy as np
import torch, torchvision

from ..trainingtask import TrainingTask
from ..datalib import random_crop, random_rotate_flip, Color, convert_rgb_to_mask

SegmentationBatch = tp.Tuple[torch.Tensor, torch.Tensor]

class SegmentationTask(TrainingTask):
    def __init__(
        self, 
        *a,
        patchsize:      int,
        colors:         tp.List[Color],
        ignore_colors:  tp.List[Color]|None    = None,
        cropfactors:    tp.Tuple[float, float] = (0.75, 1.33),
        pos_weight:     float                  = 1.0,
        **kw
    ):
        assert len(colors) == 1, NotImplementedError('TODO: implement multiclass training')
        super().__init__(*a, **kw)
        self.patchsize     = patchsize
        self.cropfactors   = cropfactors
        self.colors        = colors
        self.ignore_colors = ignore_colors
        self.pos_weight    = torch.as_tensor(pos_weight, dtype=torch.float32)
    
    def training_step(self, batch:SegmentationBatch) -> tp.Tuple[torch.Tensor, tp.Dict]:
        x,t     = batch
        assert len(t.shape) == 4 and t.shape[1] == 3, 'Expecting batched RGB targets'
        assert len(x.shape) == 4 and x.shape[1] == 3, 'Expecting batched RGB inputs'

        #x,t     = self._batch_to_device((x,t))
        x,t     = self.augment((x,t))
        
        t_rgb   = t
        t       = convert_rgb_to_mask(t_rgb, self.colors)
        weight  = None
        if self.ignore_colors is not None:
            weight = 1 - convert_rgb_to_mask(t_rgb, self.ignore_colors)
        
        y       = self.basemodule(x)
        t       = t.to(y.device)
        loss    = torch.nn.functional.binary_cross_entropy_with_logits(
            y, t, weight, pos_weight=self.pos_weight.to(y.device)
        )
        logs    = {'loss': float(loss)}
        return loss, logs
    
    def augment(self, batch:SegmentationBatch) -> SegmentationBatch:
        x,t = batch
        new_x: tp.List[torch.Tensor] = []
        new_t: tp.List[torch.Tensor] = []
        for xi,ti in zip(x,t):
            xi,ti = random_crop(
                xi, 
                ti, 
                patchsize   = self.patchsize, 
                modes       = ['bilinear', 'nearest'], 
                cropfactors = self.cropfactors
            )
            xi,ti = random_rotate_flip(xi, ti)
            new_x.append(xi)
            new_t.append(ti)
        x = torch.stack(new_x)
        t = torch.stack(new_t)
        return x,t
