import glob
import hashlib
import os
import pickle
import typing as tp

import numpy as np
import PIL.Image
import torch

from ..modellib import BaseModel
from .. import datalib
from ..datalib import random_crop, random_rotate_flip, FileTuple
from ..trainingtask import TrainingTask, Loss, Metrics
from ..paths.svg import parse_size_from_svg_string


TensorPair = datalib.TensorPair



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
            xshape_padded = (
                max(x.shape[-2], self.inputsize), 
                max(x.shape[-1], self.inputsize),
            )
            y = datalib.stitch_overlapping_patches(raw, xshape_padded, self.slack)
            y = y[..., :x.shape[-2], :x.shape[-1]]
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
    
    def _create_dataloaders(
        self, 
        DatasetClass,
        trainsplit:  tp.List,
        valsplit:    tp.List|None = None,
        trainpatchfactor: float = 2,
        ds_kw:       tp.Dict[str, tp.Any] = {},
        **ld_kw,
    ) -> tp.Tuple[tp.Iterable, tp.Iterable|None]:
        patchsize_train = patchsize_val = None
        if self.patchify:
            # NOTE: *2 for cropping during augmentation
            patchsize_train = self.inputsize * trainpatchfactor
            patchsize_val   = self.inputsize
        
        ds_train = DatasetClass(trainsplit, patchsize_train, **ds_kw)
        ld_train = datalib.create_dataloader(ds_train, shuffle=True, **ld_kw)
        ld_val   = None
        if valsplit is not None:
            ds_val = DatasetClass(valsplit, patchsize_val, **ds_kw)
            ld_val = datalib.create_dataloader(ds_val, shuffle=False, **ld_kw)
        return ld_train, ld_val

    def create_dataloaders(
        self, 
        trainsplit: tp.List, 
        valsplit:   tp.List|None = None, 
        **ld_kw,
    ) -> tp.Tuple[tp.Iterable, tp.Iterable|None]:
        return self._create_dataloaders(PatchedCachingDataset, trainsplit, valsplit, **ld_kw)
    
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
        scale:     float = 1.0,
        cachedir:  str = './cache/',
        normalize_rgb:bool = False,
    ):
        '''Dataset for image pairs. If `patchsize` is given, 
           will extract and cache patches instead of loading full images'''
        self.filepairs = filepairs
        self.scale = scale
        self.normalize_rgb = normalize_rgb
        if patchsize is not None:
            self.patchsize = patchsize
            filetuples = tp.cast(tp.List[FileTuple], filepairs)
            self.filepairs, self.grids = \
                self._cache(filetuples, cachedir=cachedir, normalize_rgb=normalize_rgb)
    
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
        filepairs:  tp.List[FileTuple], 
        prefixes:   tp.List[str] = ['in', 'an'],
        cachedir:   str  = './cache/', 
        force:      bool = False,
        normalize_rgb:bool = False,
    ):
        '''Slice data into patches and cache them into a folder.'''

        self.cachedir = cachedir = self._get_concrete_cachedir(filepairs, cachedir)
        if not force:
            patch_pairs = load_if_cached(cachedir)
            grids = load_grids_if_cached(cachedir)
            if patch_pairs is not None:
                return patch_pairs, grids
        
        print('Caching dataset into', cachedir)
        os.makedirs(cachedir, exist_ok=True)
        open(os.path.join(cachedir, '.gitignore'), 'w').write('*')

        assert_same_image_sizes(filepairs)

        lists_of_cached_files:tp.List[ tp.List[str] ] = []
        slack = max(self.patchsize // 8, 32)
        for i, prefix in enumerate(prefixes):
            files_i = [ftuple[i] for ftuple in filepairs]
            cached_files_i, cached_grids_i = slice_and_cache_images(
                cachedir,
                prefix,
                files_i,
                self.patchsize,
                slack,
                getattr(self, 'scale', 1.0),
                'bilinear' if prefix == 'in' else 'nearest',
                normalize_rgb=normalize_rgb,
            )
            lists_of_cached_files.append(cached_files_i)
        assert_same_length(*lists_of_cached_files)
        all_patch_pairs:tp.List[tp.Tuple[str,str]] = \
            list(zip(*lists_of_cached_files))
        
        cachefile = os.path.join(cachedir, 'cachefile.csv')
        datalib.save_file_tuples(cachefile, all_patch_pairs) # type: ignore

        # NOTE: assuming cached_coords are all the same
        grids_picklefile = os.path.join(cachedir, 'grids.pkl')
        pickle.dump(cached_grids_i, open(grids_picklefile, 'wb'))

        return all_patch_pairs, cached_grids_i
    
    def _get_concrete_cachedir(
        self, 
        filepairs:  tp.List[FileTuple], 
        cachedir:   str,
    ) -> str:
        hash = hashlib.sha256( 
            pickle.dumps([self.__dict__, filepairs])
        ).hexdigest()
        cachedir = os.path.join(cachedir, hash)
        cachedir = os.path.abspath(cachedir)
        return cachedir


def load_if_cached(
    cachedir:str, 
    filename:str = 'cachefile.csv', 
    n:int = 2,
    verbose:bool = True
) -> tp.List[FileTuple]|None:
    cachefile = os.path.join(cachedir, filename)
    if os.path.exists(cachefile):
        if verbose:
            print('Re-using already cached folder', cachedir)
        return datalib.load_file_tuples(cachefile, n=n, delimiter=',')
    return None

def load_grids_if_cached(
    cachedir:str, 
    filename:str = 'grids.pkl'
) -> tp.Optional[tp.List[np.ndarray]]:
    cachefile = os.path.join(cachedir, filename)
    if os.path.exists(cachefile):
        return pickle.load(open(cachefile, 'rb'))
    return None

def assert_same_length(*lists: tp.Sized) -> None:
    ''' Assert all provided iterables have the same length'''
    lengths: tp.List[int] = [len(lst) for lst in lists]
    if not lengths:
        return
    first = lengths[0]
    assert all(l == first for l in lengths), f"Lengths differ: {lengths}"


def get_image_size(path:str) -> tp.Tuple[int,int]:
    if path.endswith('.svg'):
        return parse_size_from_svg_string(open(path).read())
    else:
        return PIL.Image.open(path).size

def assert_same_image_sizes(filepairs:tp.List[FileTuple]) -> None:
    '''Assert all images a file pair/tuple have the same width and height'''
    for imtuple in filepairs:
        sizes = [get_image_size(imf) for imf in imtuple]
        if len(sizes) == 0:
            continue
        size0 = sizes[0]
        assert all(s == size0 for s in sizes), imtuple

class CachingResult:
    # all lists
    cached_files:  tp.List[str]
    originalfiles: tp.List[str]
    coordinates:   tp.List[str]


def slice_and_cache_images(
    directory:  str,
    prefix:     str,
    imagefiles: tp.List[str], 
    patchsize:  int,
    slack:      int,
    # scale factor to resize image before slicing (if not 1.0)
    scale:      float = 1.0,
    mode:       tp.Literal['nearest', 'bilinear'] = 'nearest',
    normalize_rgb: bool = False,
) -> tp.Tuple[tp.List[str], tp.List[np.ndarray]]:
    os.makedirs(directory, exist_ok=True)
    cached_files:tp.List[str] = []
    cached_coords:tp.List[np.ndarray] = []
    cached_grids: tp.List[np.ndarray] = []

    for imagefile in imagefiles:
        basename  = os.path.basename(imagefile)

        imagedata = datalib.load_image(imagefile, to_tensor=True, normalize=False)
        if normalize_rgb:
            imagedata = imagedata.permute(1,2,0) # type: ignore
            imagedata = normalize_image_from_patched_median_percentiles(imagedata)
            imagedata = imagedata.permute(2,0,1)
        imagedata = tp.cast(torch.Tensor, imagedata)
        if scale != 1.0:
            H,W = imagedata.shape[-2:]
            newshape  = ( int(H*scale), int(W*scale) )
            imagedata = datalib.resize_tensor(imagedata, newshape, mode)
        imagedata = datalib.pad_to_minimum_size(imagedata, patchsize)

        grid = datalib.grid_for_patches(imagedata.shape[-2:], patchsize, slack)
        patches = \
            datalib.slice_into_patches_from_grid(imagedata, grid)
        grid = grid.reshape(-1,4)
        for i, patch in enumerate(patches):
            patchpath = os.path.join(directory, prefix, f'{basename}.{i:03d}.png')
            cached_files.append(patchpath)
            cached_coords.append(grid[i])
            datalib.write_image_tensor(patchpath, patch)
        cached_grids.append(grid)
    return cached_files, cached_grids





def normalize_image_from_patched_median_percentiles(
    image:torch.Tensor, 
    patchsize:int = 512
) -> torch.Tensor:
    assert image.ndim == 3 and image.shape[2] == 3, image.shape
    assert image.dtype == torch.uint8
    image = image.float()

    LOs = []
    HIs = []
    for i in range(0, image.shape[0], patchsize):
        for j in range(0, image.shape[1], patchsize):
            patch = image[i:i+patchsize]
            lo = torch.quantile(patch, .01)
            hi = torch.quantile(patch, .99)

            LOs.append(lo)
            HIs.append(hi)
    
    lower = torch.median(torch.tensor(LOs))
    upper = torch.median(torch.tensor(HIs))

    image = (image * 1.0 - lower) / upper * 255
    image = torch.clip(image, 0, 255).to(torch.uint8)
    return image    
