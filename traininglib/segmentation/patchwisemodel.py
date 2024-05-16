import glob
import hashlib
import os
import pickle
import typing as tp

import torch
from ..modellib import BaseModel
from .. import datalib
from ..datalib import random_crop, random_rotate_flip
from ..trainingtask import TrainingTask, Loss, Metrics



TensorPair = tp.Tuple[torch.Tensor, torch.Tensor]



class PatchwiseModel(BaseModel):
    '''Abstract model that runs inference on patches (if patchtify==True)'''

    def __init__(self, *a, patchify:bool=False, **kw):
        super().__init__(*a, **kw)
        
        self.patchify = patchify
        if patchify:
            self.slack = max(self.inputsize // 8, 32)

    def prepare_image(self, *a, **kw) -> tp.Tuple[tp.List[torch.Tensor], torch.Tensor]:
        x, x0 = super().prepare_image(*a, **kw)
        if self.patchify:
            x = datalib.slice_into_patches_with_overlap(x[0], self.inputsize, self.slack)
        return x, x0

    def finalize_inference(   # type: ignore [override]
        self, 
        raw: tp.List[torch.Tensor], 
        x:   torch.Tensor,
    ) -> torch.Tensor:
        raw = [y.cpu() for y in raw]

        if self.patchify:
            y = datalib.stitch_overlapping_patches(raw, x.shape, self.slack)
        else:
            assert len(raw) == 1, NotImplemented
            y = raw[0]

        return y




class PatchwiseTrainingTask(TrainingTask):
    def __init__(
        self, 
        *a, 
        inputsize:   int,
        cropfactors: tp.Tuple[float, float]|None = (0.75, 1.33),
        rotate:      bool  = True,
        **kw
    ):
        super().__init__(*a, **kw)
        self.inputsize   = inputsize
        self.cropfactors = cropfactors
        self.patchify    = (cropfactors is not None)
        self.rotate      = rotate
    
    def forward_step(self, batch:TensorPair, augment:bool) -> tp.Tuple[Loss, Metrics]:
        '''Code re-use for training and validation'''
        raise NotImplementedError()

    def training_step(self, batch:TensorPair) -> tp.Tuple[Loss, Metrics]:
        return self.forward_step(batch, augment=True)
    
    def validation_step(self, batch:TensorPair) -> Metrics:
        _loss, logs = self.forward_step(batch, augment=False)
        val_logs = {f'v.{k}':v for k,v in logs.items()}
        return val_logs

    def augment(self, batch:TensorPair) -> TensorPair:
        '''Random crop, flip and rotate (if self.rotate==True)'''
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
        
        ds_train = PatchedCachingDataset(trainsplit, patchsize_train)
        ld_train = datalib.create_dataloader(ds_train, shuffle=True, **ld_kw)
        ld_val   = None
        if valsplit is not None:
            ds_val = PatchedCachingDataset(valsplit, patchsize_val)
            ld_val = datalib.create_dataloader(ds_val, shuffle=False, **ld_kw)
        return ld_train, ld_val
    
    def prepare_batch(self, raw_batch:TensorPair, augment:bool) -> TensorPair:
        x,t = raw_batch
        assert len(t.shape) == 4 and t.shape[1] == 3, 'Expecting batched RGB targets'
        assert len(x.shape) == 4 and x.shape[1] == 3, 'Expecting batched RGB inputs'
        assert t.dtype == torch.uint8
        assert x.dtype == torch.float32

        x,t = datalib.to_device(x, t, device=self.device)
        if augment:
            x,t = self.augment((x,t))
        return x,t




class PatchedCachingDataset:
    '''Dataset class that caches patches of input and annotation images'''

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
        open(os.path.join(cachedir, '.gitignore'), 'w').write('*')

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




