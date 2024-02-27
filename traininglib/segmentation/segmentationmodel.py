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
        x_patch_u8  = get_patch_from_grid(x_chw, grid, i)
        x_patch_f32 = x_patch_u8 / 255
        y_patch     = super().forward(x_patch_f32)
        y_patch     = y_patch.sigmoid()
        y           = maybe_new_y(x_chw, i, y)
        y           = paste_patch(y, y_patch, grid, i, self.slack)
        
        new_i       = i+1
        completed   = ( new_i >= grid.reshape(-1,4).size()[0] )
        y_labels    = finalize_connected_components(y, completed)

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
        )
        onnx_export.save_as_zipfile(outputfile)
        return onnx_export.onnx_bytes



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
    grid_y    = torch.arange(patchsize, H+stepsize, stepsize) # type: ignore
    grid_x    = torch.arange(patchsize, W+stepsize, stepsize) # type: ignore
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
def randperm(n:int) -> torch.Tensor:
    '''torch.randperm(n) but ONNX-exportable'''
    return torch.argsort(torch.rand(n))

@torch.jit.script_if_tracing
def pad_to_n(x:torch.Tensor, n:int) -> torch.Tensor:
    xsize = torch.as_tensor(x.size())
    H     = xsize[2]
    W     = xsize[3]
    pad_y = int( (torch.ceil(H / n).long() * n) - H )
    pad_x = int( (torch.ceil(W / n).long() * n) - W )
    x_padded = torch.nn.functional.pad(x, [0, pad_x, 0, pad_y])
    return x_padded

@torch.jit.script_if_tracing
def maxpool_3x3_strided(x:torch.Tensor, stride:int) -> torch.Tensor:
    x_padded = pad_to_n(x, n=stride)
    x_pooled = torch.nn.functional.max_pool2d(
        x_padded, kernel_size=3, stride=stride, padding=0
    )
    x_padded_size = x_padded.size()
    x_resized = torch.nn.functional.interpolate(
        x_pooled, [x_padded_size[2], x_padded_size[3]], mode='nearest'
    )
    # ones_3x3 = torch.ones(1,1,3,3)
    # x_resized = torch.nn.functional.conv_transpose2d(x_pooled, ones_3x3, stride=3)
    xsize    = x.size()
    x_sliced = x_resized[..., :xsize[2], :xsize[3]]
    return x_sliced


def connected_components_max_pool(x:torch.Tensor) -> torch.Tensor:
    '''Inefficient connected components algorithm via maxpooling'''
    assert x.dtype == torch.bool
    assert x.ndim == 4

    x = x.byte()
    n = len(x.reshape(-1))
    labeled = torch.arange(1,n+1).float()
    #shuffling makes convergence faster (in torchscript but not in onnx)
    #labeled = labeled[randperm(n)]
    labeled = labeled.reshape(x.size()) * x

    i = 0
    while 1:
        if i%2 == 0:
            update = torch.nn.functional.max_pool2d(
                labeled, kernel_size=3, stride=1, padding=1
            )
        else:
            # faster because of stride=3 but cannot use it exclusively 
            # because it will not converge
            update = maxpool_3x3_strided(labeled, stride=3)
        
        update = update * x
        change = (update != labeled)
        if not torch.any(change):
            break
        labeled = update
        i += 1
    return labeled

@torch.jit.script_if_tracing
def finalize_connected_components(
    y:         torch.Tensor, 
    completed: torch.Tensor
) -> torch.Tensor:
    assert completed.dtype == torch.bool and completed.ndim == 0
    y_labels = torch.zeros([1,1,1,1])
    if completed:
        y_binary = (y > 0.5)
        y_labels = connected_components_max_pool( y_binary )
    return y_labels

 
def connected_components_patchwise(x:torch.Tensor, patchsize:int) -> torch.Tensor:
    assert x.ndim == 4
    H,W = x.shape[-2:]
    y   = torch.zeros(x.shape, dtype=torch.int64)
    for i in range(0, H, patchsize):
        for j in range(0, W, patchsize):
            xpatch = x[..., i:, j:][...,:patchsize, :patchsize]
            ypatch = connected_components_max_pool(xpatch)
            #if i > 0:
            #   #TODO: compare/merge with patch above
            #if j > 0:
            #   #TODO: compare/merge with patch left
            y[...,i:, j:][..., :patchsize, :patchsize] = ypatch





def classmap_to_rgb(classmap:np.ndarray, colors:tp.List[Color]) -> np.ndarray:
    assert len(classmap.shape) == 2
    rgb = np.zeros( classmap.shape + (3,), dtype='uint8' )
    #zero is implicitly black
    for i,c in enumerate(colors, 1):
        rgb[classmap == i] = c
    return rgb


