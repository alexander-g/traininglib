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
from .skeletonization import skeletonize, paths_from_labeled_skeleton


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
        paths:bool                = False,
        **kw,
    ):
        super().__init__(*a, **kw)
        assert self.patchify, NotImplemented

        self.connected_components = connected_components
        self.skeletonize = skeletonize
        if paths:
            assert skeletonize and connected_components, (
                'Skeletonization and connected components required to trace paths'
            )
        self.paths = paths
    
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

        y_labels     = torch.empty([0,1,1,1], dtype=torch.int64)
        y_labels_rgb = torch.empty([1,1,3], dtype=torch.uint8)
        if self.connected_components:
            y_labels = finalize_connected_components(y, completed, self.inputsize)
            y_labels_rgb = instancemap_to_rgb(y_labels[0,0])
        
        y_skeleton = torch.empty([0,1,1,1], dtype=torch.bool)
        if self.skeletonize:
            y_skeleton = finalize_skeletonize( (y>0.5), completed )
        
        y_paths = torch.empty([0,3], dtype=torch.int64)
        if self.paths:
            y_paths = finalize_paths( y_skeleton, y_labels, completed )
        
        outputs = tuple(
            [x_u8, y]
            + ([y_labels, y_labels_rgb]   if self.connected_components else [])
            + ([y_skeleton] if self.skeletonize else [])
            + ([y_paths]    if self.paths else [])
            + [completed, new_i]
        )
        #sanity check
        assert len(self.output_names) == len(outputs)
        return outputs
    
    @property
    def output_names(self):
        return (
            ['x.output', 'y.output'] 
            + (['labels', 'labels_rgb'] if self.connected_components else [])
            + (['skeleton'] if self.skeletonize else [])
            + (['paths']    if self.paths else [])
            + ['completed', 'i.output']
        )
    
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
            outputnames  = self.output_names,
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
        outputs = self.module(
            inputfeed['x'],
            inputfeed['i'],
            inputfeed['y'],
        )
        return dict(zip(self.module.output_names, outputs))
    
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

@torch.jit.script_if_tracing
def _finalize_connected_components(
    y:         torch.Tensor, 
    completed: torch.Tensor,
    patchsize: int,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    assert completed.dtype == torch.bool and completed.ndim == 0
    y_labels     = torch.zeros([1,1,1,1], dtype=torch.int64)
    y_labels_rgb = torch.zeros([0,0,3], dtype=torch.uint8)
    if completed:
        y_binary = (y > 0.5)
        #y_labels = connected_components_max_pool( y_binary )
        y_labels = connected_components_patchwise( y_binary, patchsize )
        y_labels_rgb = instancemap_to_rgb(y_labels[0,0])
    return y_labels, y_labels_rgb

@torch.jit.script_if_tracing
def finalize_skeletonize(
    binarymap: torch.Tensor, 
    completed: torch.Tensor
) -> torch.Tensor:
    assert completed.dtype == torch.bool and completed.ndim == 0
    assert binarymap.dtype == torch.bool
    skeleton = torch.zeros([1,1,1,1], dtype=torch.bool)
    if completed:
        skeleton = skeletonize(binarymap)
    return skeleton


@torch.jit.script_if_tracing
def finalize_paths(
    skeletonmap: torch.Tensor,
    labelmap:    torch.Tensor,
    completed:   torch.Tensor,
) -> torch.Tensor:
    assert completed.dtype == torch.bool and completed.ndim == 0
    assert skeletonmap.dtype == torch.bool
    assert labelmap.dtype    == torch.int64
    assert skeletonmap.shape == labelmap.shape
    assert skeletonmap.ndim  == 4
    assert skeletonmap.shape[:2] == (1,1)

    paths = torch.empty([0,3], dtype=torch.int64)
    if completed:
        labeled_skeleton = skeletonmap * labelmap
        paths = paths_from_labeled_skeleton(labeled_skeleton[0,0])
    return paths



def classmap_to_rgb(classmap:np.ndarray, colors:tp.List[Color]) -> np.ndarray:
    assert len(classmap.shape) == 2
    rgb = np.zeros( classmap.shape + (3,), dtype='uint8' )
    #zero is implicitly black
    for i,c in enumerate(colors, 1):
        rgb[classmap == i] = c
    return rgb


def _pseudorandom_hue(i:torch.Tensor) -> torch.Tensor:
    '''Map an integer to range 0..360 with a minimum distance 
       inbetween consecutive integers.'''
    assert i.dtype == torch.int64
    return i * 360//4 % 355

def _pseudorandom_saturation(i:torch.Tensor) -> torch.Tensor:
    '''Map an integer to range 0.4..0.7 with a minimum distance 
       inbetween consecutive integers.'''
    assert i.dtype == torch.int64
    return i * 0.3/5 % 0.28 + 0.4

def hsv_to_rgb(h:torch.Tensor, s:torch.Tensor, v:torch.Tensor) -> torch.Tensor:
    '''HSV to RGB alternative conversion formula from wikipedia'''
    assert h.dtype == torch.int64   # range 0..360
    assert s.dtype == torch.float32 # range 0..1
    assert v.dtype == torch.float32 # range 0..1
    k_r = (5 + h/60) % 6
    k_g = (3 + h/60) % 6
    k_b = (1 + h/60) % 6

    r = v - v*s*torch.min(k_r, 4-k_r).clip(0,1)
    g = v - v*s*torch.min(k_g, 4-k_g).clip(0,1)
    b = v - v*s*torch.min(k_b, 4-k_b).clip(0,1)
    return torch.stack([r,g,b])

@torch.jit.script_if_tracing
def instancemap_to_rgb(instancemap:torch.Tensor) -> torch.Tensor:
    '''Colorize a map containing unique integer labels for instances
       #TODO: slow in onnx '''
    assert instancemap.dtype == torch.int64
    assert instancemap.ndim == 2

    rgbmap = torch.zeros( instancemap.shape + (3,), dtype=torch.float32 )
    #zero is implicitly black
    labels = torch.unique(instancemap)
    labels = labels[labels!=0]
    for i,l in enumerate(labels):
        mask = (instancemap == l)
        hue  = _pseudorandom_hue(torch.tensor(i))
        sat  = _pseudorandom_saturation(torch.tensor(i))
        rgb  = hsv_to_rgb(hue, sat, torch.tensor(1.0))
        # NOTE: onnx error
        #rgbmap[mask] = (rgb*255).to(torch.uint8)
        rgbmap[mask] += rgb
    return (rgbmap*255).to(torch.uint8)


