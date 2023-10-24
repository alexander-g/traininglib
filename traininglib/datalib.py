import typing as tp
import os
import glob
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
    mode: tp.Literal['nearest', 'bilinear'],
    align_corners: bool|None = None,
) -> torch.Tensor:
    assert torch.is_tensor(x)
    assert len(x.shape) in [3,4]
    x0 = x
    if len(x0.shape) == 3:
        x = x[np.newaxis]
    y = torch.nn.functional.interpolate(x, size, mode=mode, align_corners=align_corners)
    if len(x0.shape) == 3:
        y = y[0]
    return y

def write_image(filepath:str, x:torch.Tensor, makedirs:bool=True) -> None:
    assert torch.is_tensor(x)
    assert x.dtype == torch.float32
    assert x.min() >= 0 and x.max() <= 1
    assert len(x.shape) == 3 and x.shape[0] == 3

    if makedirs:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    PIL.Image.fromarray(
        (x.cpu().detach().numpy().transpose(1,2,0) * 255).astype('uint8')
    ).save(filepath)


Color = tp.Tuple[int,int,int]

def convert_rgb_to_mask(
    rgbdata:    torch.Tensor,
    colors:     tp.List[Color], 
) -> torch.Tensor:
    '''Convert a RGB image (HWC u8) to a binary mask from the specified colors'''
    assert rgbdata.dtype == torch.uint8, rgbdata.dtype
    assert rgbdata.shape[1] == 3, rgbdata.shape
    #convert BCHW to BHWC
    rgbdata       = rgbdata.moveaxis(1, -1)
    #convert to single-channel uint32 for faster processsing
    rgbadata      = torch.cat([rgbdata, 255*torch.ones_like(rgbdata[...,:1])], dim=-1)
    data_uint32   = rgbadata.view(torch.int32)[...,0]
    colors_uint32 = torch.as_tensor(
        [c+(255,) for c in colors], dtype=torch.uint8
    ).view(torch.int32).to(data_uint32.device)
    mask          = torch.isin(data_uint32, colors_uint32).float()
    return mask[:,None]


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

def collect_inputfiles(splitfile_or_glob:str, *a, **kw) -> tp.List[str]:
    '''Return a list of input images, either from a csv split file or by expanding a glob'''
    if splitfile_or_glob.endswith('.csv'):
        pairs  = load_file_pairs(splitfile_or_glob)
        inputs = [i for i,a in pairs]
        return inputs
    else:
        return glob.glob(splitfile_or_glob)


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

