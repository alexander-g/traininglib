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
        *,
        ds_kw:        tp.Dict[str, tp.Any]        = {},
        ld_kw:        tp.Dict[str, tp.Any]        = {},
        task_kw:      tp.Dict[str, tp.Any]        = {},
        fit_kw:       tp.Dict[str, tp.Any]        = {},
    ):
        colors  = [c.color for c in self.classes]
        task_kw = {'colors': colors, 'patchsize':self.inputsize} | task_kw
        ds_kw   = {'patchsize':self.inputsize*2 if self.patchify else None} | ds_kw
        return super()._start_training(
            trainsplit, 
            SegmentationDataset, 
            SegmentationTask, 
            task_kw = task_kw, 
            ds_kw   = ds_kw, 
            ld_kw   = ld_kw,
            fit_kw  = fit_kw
        )

def classmap_to_rgb(classmap:np.ndarray, colors:tp.List[Color]) -> np.ndarray:
    assert len(classmap.shape) == 2
    rgb = np.zeros( classmap.shape + (3,), dtype='uint8' )
    #zero is implicitly black
    for i,c in enumerate(colors, 1):
        rgb[classmap == i] = c
    return rgb





class SegmentationDataset:
    def __init__(self, filepairs:tp.List[tp.Tuple[str,str]], patchsize:int|None):
        '''Dataset for image pairs. If `patchsize` is given, 
           will extract and cache patches instead of loading full images'''
        self.filepairs = filepairs
        if patchsize is not None:
            self.patchsize = patchsize
            self.filepairs = self._cache(filepairs)
    
    def __len__(self) -> int:
        return len(self.filepairs)
    
    def __getitem__(self, i:int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        inputfile, annotationfile = self.filepairs[i]
        inputdata  = datalib.load_image(inputfile, to_tensor=True)
        annotation = datalib.load_image(annotationfile, normalize=False, to_tensor=True)
        
        #to make mypy happy
        inputdata  = tp.cast(torch.Tensor, inputdata)
        annotation = tp.cast(torch.Tensor, annotation)
        return inputdata, annotation
    
    def _cache(
        self, 
        filepairs:  tp.List[tp.Tuple[str,str]], 
        cachedir:   str                         = './cache/', 
        force:      bool                        = False
    ):
        '''Slice data into patches and cache them into a folder.'''

        hash = hashlib.sha256( pickle.dumps([self, filepairs]) ).hexdigest()
        cachedir = os.path.join(cachedir, hash)
        if os.path.exists(cachedir) and not force:
            print('Re-using already cached folder', cachedir)
            in_paths = sorted(glob.glob(os.path.join(cachedir, 'in', '*.png')))
            an_paths = sorted(glob.glob(os.path.join(cachedir, 'an', '*.png')))
            assert len(in_paths) == len(an_paths)
            patch_pairs = list(zip(in_paths, an_paths))
            return patch_pairs
        
        print('Caching dataset into', cachedir)
        os.makedirs(cachedir, exist_ok=True)
        slack = 32
        all_patch_pairs:tp.List[tp.Tuple[str,str]] = []
        for inputfile, annotationfile in filepairs:
            in_data = datalib.load_image(inputfile, to_tensor=True)
            in_data = tp.cast(torch.Tensor, in_data)
            an_data = datalib.load_image(annotationfile, to_tensor=True)
            an_data = tp.cast(torch.Tensor, an_data)
            in_patches = datalib.slice_into_patches_with_overlap(in_data, self.patchsize, slack)
            an_patches = datalib.slice_into_patches_with_overlap(an_data, self.patchsize, slack)
            for i, (in_patch, an_patch) in enumerate(zip(in_patches, an_patches)):
                in_patchpath = os.path.join(
                    cachedir, 'in', f'{os.path.basename(inputfile)}.{i:03d}.png'
                )
                an_patchpath = os.path.join(
                    cachedir, 'an', f'{os.path.basename(annotationfile)}.{i:03d}.png'
                )
                all_patch_pairs.append((in_patchpath, an_patchpath))

                datalib.write_image_tensor(in_patchpath, in_patch)
                datalib.write_image_tensor(an_patchpath, an_patch)
        return all_patch_pairs

