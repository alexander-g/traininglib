import typing as tp
import glob
import hashlib
import os
import pickle
import numpy as np
import torch, torchvision

from ..trainingtask import TrainingTask, Loss, Metrics
from .. import datalib
from ..datalib import random_crop, random_rotate_flip, Color, convert_rgb_to_mask

SegmentationBatch = tp.Tuple[torch.Tensor, torch.Tensor]

class SegmentationTask(TrainingTask):
    def __init__(
        self, 
        *a,
        inputsize:      int,
        colors:         tp.List[Color],
        ignore_colors:  tp.List[Color]|None         = None,
        cropfactors:    tp.Tuple[float, float]|None = (0.75, 1.33),
        pos_weight:     float = 1.0,
        margin_weight:  float = 0.0,
        rotate:         bool  = True,
        **kw
    ):
        assert len(colors) == 1, NotImplementedError('TODO: implement multiclass training')
        super().__init__(*a, **kw)
        self.inputsize     = inputsize
        self.cropfactors   = cropfactors
        self.patchify      = (cropfactors is not None)
        self.colors        = colors
        self.ignore_colors = ignore_colors
        self.pos_weight    = torch.as_tensor(pos_weight, dtype=torch.float32)
        self.margin_weight = margin_weight
        self.rotate        = rotate
    

    def forward_step(self, batch:SegmentationBatch, augment:bool) -> tp.Tuple[Loss, Metrics]:
        '''Code re-use for training and validation'''
        batch = datalib.to_device(*batch, device=self.device)
        x,t = batch
        assert len(t.shape) == 4 and t.shape[1] == 3, 'Expecting batched RGB targets'
        assert len(x.shape) == 4 and x.shape[1] == 3, 'Expecting batched RGB inputs'

        if augment:
            x,t = self.augment((x,t))
        
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

    def training_step(self, batch:SegmentationBatch) -> tp.Tuple[Loss, Metrics]:
        return self.forward_step(batch, augment=True)
    
    def validation_step(self, batch:SegmentationBatch) -> Metrics:
        _loss, logs = self.forward_step(batch, augment=False)
        val_logs = {f'v.{k}':v for k,v in logs.items()}
        return val_logs
    
    def augment(self, batch:SegmentationBatch) -> SegmentationBatch:
        x,t = batch
        new_x: tp.List[torch.Tensor] = []
        new_t: tp.List[torch.Tensor] = []
        for xi,ti in zip(x,t):
            if self.cropfactors is not None:
                xi,ti = random_crop(
                    xi, 
                    ti, 
                    patchsize   = self.inputsize, 
                    modes       = ['bilinear', 'nearest'], 
                    cropfactors = self.cropfactors
                )
            xi,ti = random_rotate_flip(xi, ti, rotate=self.rotate)
            new_x.append(xi)
            new_t.append(ti)
        x = torch.stack(new_x)
        t = torch.stack(new_t)
        return x,t
    
    def create_dataloaders(
        self, trainsplit:tp.List, valsplit:tp.List|None = None, **ld_kw
    ) -> tp.Tuple[tp.Iterable, tp.Iterable|None]:
        patchsize_train = patchsize_val = None
        if self.patchify:
            # NOTE: *2 for cropping during augmentation
            patchsize_train = self.inputsize * 2
            patchsize_val   = self.inputsize
        
        ds_train = SegmentationDataset(trainsplit, patchsize_train)
        ld_train = datalib.create_dataloader(ds_train, shuffle=True, **ld_kw)
        ld_val   = None
        if valsplit is not None:
            ds_val = SegmentationDataset(valsplit, patchsize_val)
            ld_val = datalib.create_dataloader(ds_val, shuffle=False, **ld_kw)
        return ld_train, ld_val


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




class SegmentationDataset:
    def __init__(self, 
        filepairs: tp.List[tp.Tuple[str,str]], 
        patchsize: int|None, 
        cachedir:  str = './cache/',
    ):
        '''Dataset for image pairs. If `patchsize` is given, 
           will extract and cache patches instead of loading full images'''
        self.filepairs = filepairs
        if patchsize is not None:
            self.patchsize = patchsize
            self.filepairs = self._cache(filepairs, cachedir)
    
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

        hash = hashlib.sha256( 
            pickle.dumps([self.__dict__, filepairs])
        ).hexdigest()
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
        slack = max(self.patchsize // 8, 32)
        all_patch_pairs:tp.List[tp.Tuple[str,str]] = []
        for inputfile, annotationfile in filepairs:
            in_data = datalib.load_image(inputfile, to_tensor=True)
            in_data = tp.cast(torch.Tensor, in_data)
            in_data = datalib.pad_to_minimum_size(in_data, self.patchsize)

            an_data = datalib.load_image(annotationfile, to_tensor=True)
            an_data = tp.cast(torch.Tensor, an_data)
            an_data = datalib.pad_to_minimum_size(an_data, self.patchsize)
            
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

