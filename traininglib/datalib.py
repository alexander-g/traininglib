import typing as tp
import os
import numpy as np
import torch, torchvision
import PIL.Image


def create_dataloader(
    ds:         torch.utils.data.Dataset, 
    batch_size: int, 
    shuffle:    bool=False, 
    num_workers:int|tp.Literal['auto'] = 'auto', 
    **kw
) -> torch.utils.data.DataLoader:
    if num_workers == 'auto':
        num_workers = os.cpu_count() or 1
    return torch.utils.data.DataLoader(
        ds, 
        batch_size, 
        shuffle, 
        collate_fn      = getattr(ds, 'collate_fn', None),
        num_workers     = num_workers, 
        pin_memory      = True,
        worker_init_fn  = lambda x: np.random.seed(torch.randint(0,1000,(1,))[0].item()+x),
        **kw
    )



def load_image(
    imagefile:str, 
    normalize:bool = True, 
    to_tensor:bool = True, 
    mode:str       = 'RGB'
) -> np.ndarray|torch.Tensor:
    '''Convenience function for loading image files'''
    image = PIL.Image.open(imagefile).convert(mode)
    if normalize and to_tensor:
        image   =   torchvision.transforms.ToTensor()(image)
    elif normalize:
        image   =   image / np.float32(255)
    elif to_tensor:
        image   =   torch.as_tensor( np.array(image) )
        if image.ndim == 2:
            image = image[..., None]
        #to CHW ordering
        image = torch.moveaxis(image, -1,0)
    else:
        image   =   np.array(image)
    return image

def ensure_imagetensor(x:str|np.ndarray) -> torch.Tensor:
    '''Convert input to a CHW image tensor (if needed).'''
    t:torch.Tensor
    if isinstance(x, str):
        _t = load_image(x, to_tensor=True)
        #make mypy happy
        t  = tp.cast(torch.Tensor, _t)
    elif not torch.is_tensor(x):
        t = torchvision.transforms.ToTensor()(x)
    return t

def resize_tensor(
    x:    torch.Tensor, 
    size: int|tp.Tuple[int,int], 
    mode: tp.Literal['nearest', 'bilinear']
) -> torch.Tensor:
    assert torch.is_tensor(x)
    assert len(x.shape) in [3,4]
    x0 = x
    if len(x0.shape) == 3:
        x = x[np.newaxis]
    y = torch.nn.functional.interpolate(x, size, mode=mode)
    if len(x0.shape) == 3:
        y = y[0]
    return y


def load_file_pairs(filepath:str, delimiter:str=',') -> tp.List[tp.Tuple[str,str]]:
    '''Load pairs of file paths from a csv file and check that they exist..'''

    lines = open(filepath, 'r').read().strip().split('\n')
    dirname = os.path.dirname(filepath)
    pairs:tp.List[tp.Tuple[str,str]] = []
    for line in lines:
        pair = [f.strip() for f in line.split(delimiter)]
        if len(pair) != 2:
            raise Exception(f'File does not contain pairs delimited by "{delimiter}"')
        #convert relative paths to absolute, starting from the textfile directory
        pair = [f if os.path.isabs(f) else os.path.join(dirname, f) for f in pair]
        if not all(os.path.exists(p) for p in pair):
            raise Exception(f'Files not found: {pair}')
        pairs.append((pair[0], pair[1]))
    return pairs


#Helper functions for slicing images (for CHW dimension ordering)
def grid_for_patches(
    imageshape:tp.Tuple[int,int]|torch.Size, patchsize:int, slack:int
) -> np.ndarray:
    H,W       = imageshape[:2]
    stepsize  = patchsize - slack
    grid      = np.stack( 
        np.meshgrid( 
            np.minimum( np.arange(patchsize, H+stepsize, stepsize), H ), 
            np.minimum( np.arange(patchsize, W+stepsize, stepsize), W ),
            indexing='ij' 
        ), 
        axis=-1 
    )
    grid      = np.concatenate([grid-patchsize, grid], axis=-1)
    grid      = np.maximum(0, grid)
    return grid

def slice_into_patches_with_overlap(
    image:torch.Tensor, patchsize:int=1024, slack:int=32
) -> tp.List[torch.Tensor]:
    image     = torch.as_tensor(image)
    grid      = grid_for_patches(image.shape[-2:], patchsize, slack)
    patches   = [image[...,i0:i1, j0:j1] for i0,j0,i1,j1 in grid.reshape(-1, 4)]
    return patches

def stitch_overlapping_patches(
    patches:        tp.List[torch.Tensor], 
    imageshape:     tp.Tuple[int,int], 
    slack:          int                     = 32, 
    out:            torch.Tensor|None       = None,
) -> torch.Tensor:
    patchsize = np.max(patches[0].shape[-2:])
    grid      = grid_for_patches(imageshape[-2:], patchsize, slack)
    halfslack = slack//2
    i0,i1     = (grid[grid.shape[0]-2,grid.shape[1]-2,(2,3)] - grid[-1,-1,(0,1)])//2
    d0 = np.stack( 
        np.meshgrid(
            [0]+[ halfslack]*(grid.shape[0]-2)+[i0]*(grid.shape[0]>1),
            [0]+[ halfslack]*(grid.shape[1]-2)+[i1]*(grid.shape[1]>1),
            indexing='ij' 
        ), 
        axis=-1
    )
    d1 = np.stack(
        np.meshgrid(     
            [-halfslack]*(grid.shape[0]-1)+[imageshape[-2]],      
            [-halfslack]*(grid.shape[1]-1)+[imageshape[-1]],
            indexing='ij'
        ), 
        axis=-1
    )
    d  = np.concatenate([d0,d1], axis=-1)
    if out is None:
        out = torch.empty(patches[0].shape[:-2] + imageshape[-2:], dtype=patches[0].dtype)
    for patch,gi,di in zip(patches, d.reshape(-1,4), (grid+d).reshape(-1,4)):
        out[...,di[0]:di[2], di[1]:di[3]] = patch[...,gi[0]:gi[2], gi[1]:gi[3]]
    return out

