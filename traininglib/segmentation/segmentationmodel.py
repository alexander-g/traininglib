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


StateDict = tp.Dict[str, torch.Tensor]

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
        y = raw
        if y.shape[2] != x.shape[2] or y.shape[3] != x.shape[3]:
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
    

class SegmentationModel2(SegmentationModel):    
    def forward(  # type: ignore[override]
        self, 
        x: torch.Tensor, 
        i: torch.Tensor,
        y: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        inputsize = image_size(x)
        grid      = grid_for_patches(inputsize, self.inputsize, self.slack)
        x_patch   = get_patch_from_grid(x, grid, i)
        y_patch   = super().forward(x_patch)
        y         = paste_patch(y, y_patch, grid, i, self.slack)
        
        completed = ( i+1 >= grid.reshape(-1,4).size()[0] )
        return y, completed

@torch.jit.script
def image_size(x:torch.Tensor) -> torch.Tensor:
    '''Height and width of a (B,C,H,W) tensor, dynamic even when tracing'''
    assert x.ndim == 4
    return torch.as_tensor(x.size()[-2:])

@torch.jit.script
def grid_for_patches(
    imageshape: torch.Tensor,
    patchsize:  int, 
    slack:      int
) -> torch.Tensor:
    assert imageshape.ndim == 1 and imageshape.shape[0] == 2

    H,W       = torch.tensor(imageshape[0]), torch.tensor(imageshape[1])
    stepsize  = patchsize - slack
    grid      = torch.stack( 
        torch.meshgrid( 
            torch.minimum( torch.arange(patchsize, H+stepsize, stepsize), H ), 
            torch.minimum( torch.arange(patchsize, W+stepsize, stepsize), W ),
            indexing='ij' 
        ), 
        dim = -1 
    )
    grid      = torch.concatenate([grid-patchsize, grid], dim=-1)
    grid      = torch.maximum(torch.tensor(0.0), grid)
    return grid

@torch.jit.script
def get_patch_from_grid(
    x:    torch.Tensor, 
    grid: torch.Tensor, 
    i:    torch.Tensor
) -> torch.Tensor:
    patch = grid.reshape(-1, 4)[i].long()
    return x[..., patch[0]:patch[2], patch[1]:patch[3]]

# @torch.jit.script
# def get_patch(x:torch.Tensor, i:torch.Tensor, patchsize:int, slack:int) -> torch.Tensor:
#     xsize = image_size(x)
#     grid  = grid_for_patches(xsize, patchsize, slack).reshape(-1, 4)
#     return get_patch_from_grid(x, grid, i)


@torch.jit.script
def paste_patch(
    output: torch.Tensor, 
    patch:  torch.Tensor, 
    grid:   torch.Tensor, 
    i:      torch.Tensor,
    slack:  int,
) -> torch.Tensor:
    assert output.ndim == 4, output.shape
    assert grid.ndim == 3, grid.shape

    output[:,:,:patch.shape[2], :patch.shape[3]] = patch

    imageshape = image_size(output)
    halfslack  = torch.as_tensor(slack//2)
    grid_h, grid_w = grid.shape[0], grid.shape[1]
    #the last grid patches might overlap larger than the previous ones
    last_halfslack = (grid[grid_h-2, grid_w-2, 2:4] - grid[-1,-1,0:2])//2
    last_hslack0   = last_halfslack[0]
    last_hslack1   = last_halfslack[1]

    crop_top    = torch.cat(
        [torch.tensor([0]), halfslack.repeat(grid_h-2), last_hslack0.repeat(1)]
    )
    crop_left   = torch.cat(
        [torch.tensor([0]), halfslack.repeat(grid_w-2), last_hslack1.repeat(1)]
    )
    crop_bottom = torch.cat([-halfslack.repeat(grid_h-1), imageshape[-2][None]])
    crop_right  = torch.cat([-halfslack.repeat(grid_w-1), imageshape[-1][None]])

    d0 = torch.stack( 
        torch.meshgrid(crop_top, crop_left, indexing='ij'), dim=-1
    )
    d1 = torch.stack(
        torch.meshgrid(crop_bottom, crop_right, indexing='ij'), dim=-1
    )
    d  = torch.cat([d0,d1], dim=-1)
    di = d.reshape(-1,4)[i].long()
    gi = (grid + d).reshape(-1,4)[i].long()
    output[...,gi[0]:gi[2], gi[1]:gi[3]] = patch[...,di[0]:di[2], di[1]:di[3]]
    return output



def classmap_to_rgb(classmap:np.ndarray, colors:tp.List[Color]) -> np.ndarray:
    assert len(classmap.shape) == 2
    rgb = np.zeros( classmap.shape + (3,), dtype='uint8' )
    #zero is implicitly black
    for i,c in enumerate(colors, 1):
        rgb[classmap == i] = c
    return rgb


