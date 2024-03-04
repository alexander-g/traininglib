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
from .connectedcomponents import (
    connected_components_max_pool,
    connected_components_patchwise
)


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
        module:             torch.nn.Module|None = None,
        weights:            str|None        = None,
        weights_backbone:   str|None        = None,
    ):
        assert weights_backbone is None, NotImplemented
        if module is None:
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
   

class SegmentationModel_ONNX(SegmentationModel):
    '''SegmentationModel that processes only a patch of the input at a time.
       Exportable for ONNX inference. '''
    
    def __init__(
        self,
        *a,
        connected_components:bool = False,
        skeletonize:bool          = False,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.connected_components = connected_components
        self.skeletonize = skeletonize
    
    def forward(  # type: ignore[override]
        self, 
        x_u8: torch.Tensor, 
        i:    torch.Tensor,
        y:    torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, ...]:
        assert x_u8.dtype == torch.uint8, 'uint8 data expected for onnx'
        assert x_u8.shape[3] == 3, 'RGB data expected for onnx'
        x_chw = (x_u8).permute(0,3,1,2)

        inputsize   = image_size(x_chw)
        grid        = grid_for_patches(inputsize, self.inputsize, self.slack)
        for i in torch.arange(i, i+1): # type: ignore
            x_patch_u8  = get_patch_from_grid(x_chw, grid, i)
            x_patch_f32 = x_patch_u8 / 255
            y_patch     = super().forward(x_patch_f32)
            y_patch     = y_patch.sigmoid()
            #y_patch     = (y_patch > 0.5)
            y           = maybe_new_y(x_chw, i, y)
            y           = paste_patch(y, y_patch, grid, i, self.slack)
        
        new_i       = i+1
        completed   = ( new_i >= grid.reshape(-1,4).size()[0] )
        y_labels    = y
        if self.connected_components:
            y_labels = finalize_connected_components(y, completed, self.inputsize)
        if self.skeletonize:
            #TODO
            pass

        return x_u8, y, y_labels, completed, new_i
    
    def export_to_onnx(self, outputfile:str) -> bytes:
        from traininglib import onnxlib
        args = {
            'x': torch.ones([1,1024,1024,3], dtype=torch.uint8),
            'i': torch.tensor(0),
            'y': torch.zeros([1,1,1,1]),
        }
        onnx_export = onnxlib.export_model_inference(
            model        = self, 
            inputs       = tuple(args.values()), 
            inputnames   = list(args.keys()), 
            outputnames  = ['x.output', 'y.output', 'labels', 'completed', 'i.output'],
            dynamic_axes = {'x':[1,2], 'y':[0,1,2,3]},
            export_params = True,
        )
        onnx_export.save_as_zipfile(outputfile)
        return onnx_export.onnx_bytes



TensorDict = tp.Dict[str, torch.Tensor]

class SegmentationModel_TorchScript(torch.nn.Module):
    def __init__(self, module:SegmentationModel_ONNX):
        super().__init__()
        self.module = module
    
    def forward(self, inputfeed:TensorDict) -> TensorDict:
        x_u8, y, y_labels, completed, new_i = self.module(
            inputfeed['x'],
            inputfeed['i'],
            inputfeed['y'],
        )
        return {
            'x':         x_u8,
            'y':         y,
            'labels':    y_labels,
            'completed': completed,
            'i':         new_i,
        }
    
    def export_to_torchscript(self, outputfile:str) -> None:
        from traininglib import torchscriptlib
        args = {
            'x_u8': torch.ones([1,1024,1024,3], dtype=torch.uint8),
            'i':    torch.tensor(0),
            'y':    torch.zeros([1,1,1,1]),
        }
        args['x'] = args['x_u8']
        traced = torch.jit.trace(self, args, strict=False)
        #traced.save(outputfile)
        exported = torchscriptlib.ExportedTorchScript(
            torchscriptmodule = traced,
            modulestate       = {},
            name              = 'inference',
        )
        del args['x_u8']
        exported.save_as_zipfile(outputfile, args)





@torch.jit.script_if_tracing
def image_size(x:torch.Tensor) -> torch.Tensor:
    '''Height and width of a (B,C,H,W) tensor, dynamic even when tracing'''
    assert x.ndim == 4
    return torch.as_tensor(x.size()[-2:])

@torch.jit.script_if_tracing
def maybe_new_y(
    x:torch.Tensor, 
    i:torch.Tensor, 
    y:torch.Tensor
) -> torch.Tensor:
    if i > 0:
        return y
    return torch.zeros(x[:,:1].shape, dtype=torch.float32, device=x.device)

@torch.jit.script_if_tracing
def grid_for_patches(
    imageshape: torch.Tensor,
    patchsize:  int, 
    slack:      int
) -> torch.Tensor:
    assert imageshape.ndim == 1 and imageshape.shape[0] == 2

    H,W       = torch.tensor(imageshape[0]), torch.tensor(imageshape[1])
    stepsize  = patchsize - slack
    grid_y    = torch.arange(int(patchsize), int(H+stepsize), int(stepsize))
    grid_x    = torch.arange(int(patchsize), int(W+stepsize), int(stepsize))
    grid      = torch.stack( 
        torch.meshgrid( 
            torch.minimum( grid_y, H ), 
            torch.minimum( grid_x, W ),
            indexing='ij' 
        ), 
        dim = -1 
    )
    grid      = torch.concatenate([grid-patchsize, grid], dim=-1)
    grid      = torch.maximum(torch.tensor(0.0), grid)
    return grid

@torch.jit.script_if_tracing
def get_patch_from_grid(
    x:    torch.Tensor, 
    grid: torch.Tensor, 
    i:    torch.Tensor
) -> torch.Tensor:
    patch = grid.reshape(-1, 4)[i].long()
    return x[..., patch[0]:patch[2], patch[1]:patch[3]]



@torch.jit.script_if_tracing
def paste_patch(
    output: torch.Tensor, 
    patch:  torch.Tensor, 
    grid:   torch.Tensor, 
    i:      torch.Tensor,
    slack:  int,
) -> torch.Tensor:
    assert output.ndim == 4, output.shape

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

@torch.jit.script_if_tracing
def finalize_connected_components(
    y:         torch.Tensor, 
    completed: torch.Tensor,
    patchsize: int,
) -> torch.Tensor:
    assert completed.dtype == torch.bool and completed.ndim == 0
    y_labels = torch.zeros([1,1,1,1], dtype=torch.int64)
    if completed:
        y_binary = (y > 0.5)
        #y_labels = connected_components_max_pool( y_binary )
        y_labels = connected_components_patchwise( y_binary, patchsize )
    return y_labels



def classmap_to_rgb(classmap:np.ndarray, colors:tp.List[Color]) -> np.ndarray:
    assert len(classmap.shape) == 2
    rgb = np.zeros( classmap.shape + (3,), dtype='uint8' )
    #zero is implicitly black
    for i,c in enumerate(colors, 1):
        rgb[classmap == i] = c
    return rgb


