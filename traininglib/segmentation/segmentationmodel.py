import typing as tp
import os
import glob
import hashlib
import pickle
import numpy as np
import torch

from .. import datalib
from ..datalib import Color
from ..unet import UNet
from ..modellib import BaseModel, load_weights
from .segmentationtraining import SegmentationTask


class Class(tp.NamedTuple):
    name:  str
    color: Color

class SegmentationOutput(tp.NamedTuple):
    raw:      tp.Any
    classmap: np.ndarray
    rgb:      np.ndarray



class SegmentationModel(BaseModel):
    def __init__(
        self, 
        inputsize:          int, 
        classes:            tp.List[Class],
        patchify:           bool|None       = None,
        weights:            str|None        = None,
        weights_backbone:   str|None        = None,
    ):
        assert weights_backbone is None, NotImplemented
        module = UNet(output_channels=len(classes))
        super().__init__(module, inputsize)
        if weights is not None:
            load_weights(weights, self)
        
        self.classes   = classes
        self.patchify  = patchify
        if patchify:
            self.slack = max(self.inputsize // 8, 32)

    def postprocess(self, raw:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        y = datalib.resize_tensor(raw, x.shape[-2:], "bilinear")
        return y
    
    def prepare_image(self, *a, **kw) -> tp.Tuple[tp.List[torch.Tensor], torch.Tensor]:
        x, x0 = super().prepare_image(*a, **kw)
        if self.patchify:
            x = datalib.slice_into_patches_with_overlap(x[0], self.inputsize, self.slack)
        return x, x0
    
    def finalize_inference(   # type: ignore [override]
        self, raw:tp.List[torch.Tensor], x:torch.Tensor
    ) -> SegmentationOutput:
        raw = [y.cpu() for y in raw]

        if self.patchify:
            y = datalib.stitch_overlapping_patches(raw, x.shape, self.slack)
        else:
            assert len(raw) == 1, NotImplemented
            y = raw[0]
        
        if y.shape[1] == 1:
            classmap = torch.sigmoid(y)[0,0].numpy()
            classmap = (classmap >= 0.5).astype('float32')
        else:
            classmap = torch.argmax(y, dim=1).numpy()
        
        colors = [cls.color for cls in self.classes]
        return SegmentationOutput(
            raw         = y,
            classmap    = classmap,
            rgb         = classmap_to_rgb(classmap, colors),
        )
    
    def start_training(
        self,
        trainsplit: tp.List[tp.Tuple[str,str]],
        valsplit:   tp.List[tp.Tuple[str,str]] | None = None,
        *,
        task_kw:    tp.Dict[str, tp.Any] = {},
        fit_kw:     tp.Dict[str, tp.Any] = {},
    ):
        colors  = [c.color for c in self.classes]
        task_kw = {'colors': colors, 'inputsize':self.inputsize} | task_kw
        return super()._start_training(
            SegmentationTask, 
            trainsplit, 
            valsplit,
            task_kw = task_kw, 
            fit_kw  = fit_kw
        )

def classmap_to_rgb(classmap:np.ndarray, colors:tp.List[Color]) -> np.ndarray:
    assert len(classmap.shape) == 2
    rgb = np.zeros( classmap.shape + (3,), dtype='uint8' )
    #zero is implicitly black
    for i,c in enumerate(colors, 1):
        rgb[classmap == i] = c
    return rgb


