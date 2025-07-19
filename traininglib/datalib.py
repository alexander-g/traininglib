import typing as tp
import os
import glob
import numpy as np
import torch, torchvision
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

from .dataloader import ThreadedDataLoader


FilePair   = tp.Tuple[str,str]
FileTriple = tp.Tuple[str,str,str]
FileTuple  = tp.Tuple[str,...]

TensorPair = tp.Tuple[torch.Tensor, torch.Tensor]

#width first
ImageSize = tp.Tuple[int,int]


def create_dataloader(
    dataset, 
    batch_size: int, 
    shuffle:    bool=False, 
    num_workers:int|tp.Literal['auto'] = 'auto', 
    loader_type:tp.Literal['torch', 'threaded'] = 'torch',
    **kw
) -> torch.utils.data.DataLoader:
    assert loader_type in ['torch', 'threaded']

    if num_workers == 'auto':
        num_workers = min(os.cpu_count() or 1, batch_size)
    
    collate_fn = getattr(dataset, 'collate_fn', None)

    if loader_type == 'torch':
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size, 
            shuffle, 
            collate_fn      = collate_fn,
            num_workers     = num_workers, 
            pin_memory      = False, # getting out-of-memory sometimes if True
            #persistent_workers = True,
            worker_init_fn  = _worker_init_fn,
            **kw
        )
    elif loader_type == 'threaded':
        return ThreadedDataLoader(
            dataset,
            batch_size,
            collate_fn,
            num_workers,
            shuffle
        )

def _worker_init_fn(x):
    np.random.seed(torch.randint(0,1000,(1,))[0].item()+x)


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

def ensure_imagetensor(x:str|np.ndarray|torch.Tensor, **kw) -> torch.Tensor:
    '''Convert input to a CHW image tensor (if needed).'''
    t:torch.Tensor
    if isinstance(x, str):
        _t = load_image(x, to_tensor=True, **kw)
        #make mypy happy
        t  = tp.cast(torch.Tensor, _t)
    elif not torch.is_tensor(x):
        t = torchvision.transforms.ToTensor()(x).float()
    else:
        t = tp.cast(torch.Tensor, x)
    return t

def resize_tensor(
    x:    torch.Tensor, 
    size: int|tp.Tuple[int,int]|torch.Size, 
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

def to_device(*x:torch.Tensor, device:torch.device|str) -> tp.List[torch.Tensor]:
    return [xi.to(device) for xi in x]

interpolation_modes = {
    'nearest':  torchvision.transforms.InterpolationMode.NEAREST,
    'bilinear': torchvision.transforms.InterpolationMode.BILINEAR,
}

#TODO: get torchvision.transforms.v2 to work
def random_crop(
    *x:         torch.Tensor,
    patchsize:  int, 
    modes:      tp.List[tp.Literal['nearest', 'bilinear']],
    cropfactors:tp.Tuple[float, float] = (0.75, 1.33),
) -> tp.List[torch.Tensor]:
    '''Perform random crops on multiple (BCHW) tensors'''
    # TODO: generate coordinates for every image in batch (not list ofc)
    H,W = x[0].shape[-2:]
    lo  = patchsize * cropfactors[0]
    hi  = patchsize * cropfactors[1]
    h   = int(lo + torch.rand(1) * (hi-lo))
    w   = int(lo + torch.rand(1) * (hi-lo))
    y0  = int(torch.rand(1) * (H - h))
    x0  = int(torch.rand(1) * (W - w))

    output = list(x)
    for i in range(len(output)):
        output[i] = torchvision.transforms.functional.resized_crop(
            output[i], y0, x0, h, w, [patchsize]*2, interpolation_modes[modes[i]]
        )
    return output


def random_rotate_flip(
    *x:torch.Tensor, 
    flip:bool   = True, 
    rotate:bool = True,
) -> tp.List[torch.Tensor]:
    '''Perform random rotation and flip operations on multiple (BCHW) tensors'''
    output = list(x)
    if rotate:
        k = np.random.randint(4)
        for i in range(len(output)):
            output[i] = torch.rot90(output[i], k, dims=(-2,-1))
    if flip and np.random.random() < 0.5:
        for i in range(len(output)):
            output[i] = torch.flip(output[i], dims=(-1,))
    return output

def write_image(filepath:str, x:np.ndarray, makedirs:bool=True) -> None:
    assert (len(x.shape) == 3 and x.shape[-1] == 3) or len(x.shape) == 2, x.shape
    if x.dtype in [np.float32, np.float64, torch.float32]:
        x = (x * 255).astype('uint8')
    
    if makedirs:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    PIL.Image.fromarray(x).save(filepath)

def write_image_tensor(filepath:str, x:torch.Tensor, makedirs:bool=True) -> None:
    assert torch.is_tensor(x)
    assert (len(x.shape) == 3 and x.shape[0] == 3) or len(x.shape) == 2, x.shape

    x_np = x.cpu().detach().numpy()
    if len(x.shape) == 3:
        x_np = x_np.transpose(1,2,0)
    return write_image(filepath, x_np, makedirs)


def pad_to_minimum_size(
    x: torch.Tensor, 
    minsize: int, 
    center:  bool=False,
) -> torch.Tensor:
    '''If necessary pad the input on the last two dimensions to at least `minsize`
       If center is True, the original image is centered in the padded output.'''
    assert x.ndim >= 2, x.shape
    H,W = x.shape[-2:]
    pad_h = max(0, minsize - H)
    pad_w = max(0, minsize - W)
    
    if center:
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
    else:
        pad_top, pad_left = 0, 0
        pad_bottom, pad_right = pad_h, pad_w
    
    paddings = (pad_left, pad_right, pad_top, pad_bottom)
    return torch.nn.functional.pad(x, paddings)



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


def load_file_tuples(filepath:str, delimiter:str, n:int) -> tp.List[FileTuple]:
    '''Load n-tuples of file paths from a csv file and check that they exist.'''
    lines = open(filepath, 'r').read().strip().split('\n')
    dirname = os.path.dirname(filepath)
    pairs:tp.List[FileTuple] = []
    for line in lines:
        pair = [f.strip() for f in line.split(delimiter)]
        if len(pair) != n:
            raise Exception(
                f'File does not contain {n}-tuples delimited by "{delimiter}"'
            )
        #convert relative paths to absolute, starting from the textfile directory
        pair = [f if os.path.isabs(f) else os.path.join(dirname, f) for f in pair]
        if not all(os.path.exists(p) for p in pair):
            raise Exception(
                f'Files not found: {[f for f in pair if not os.path.exists(f)]}'
            )
        pairs.append(tuple(pair))
    return pairs

def load_file_triples(filepath:str, delimiter:str=',') -> tp.List[FileTriple]:
    '''Load triples of file triples from a csv file and check that they exist.'''
    pairs = load_file_tuples(filepath, delimiter, n=3)
    return tp.cast(tp.List[FileTriple], pairs)

def load_file_pairs(filepath:str, delimiter:str=',') -> tp.List[FilePair]:
    '''Load pairs of file paths from a csv file and check that they exist.'''
    pairs = load_file_tuples(filepath, delimiter, n=2)
    return tp.cast(tp.List[FilePair], pairs)


def collect_inputfiles(splitfile_or_glob:str, *a, **kw) -> tp.List[str]:
    '''Return a list of input images, either from a csv split file or by expanding a glob'''
    if splitfile_or_glob.endswith('.csv') or splitfile_or_glob.endswith('.txt'):
        pairs  = load_file_pairs(splitfile_or_glob)
        inputs = [i for i,a in pairs]
        return inputs
    else:
        return sorted(glob.glob(splitfile_or_glob))

def save_file_tuples(filepath:str, filetuples:tp.List[FileTuple], delimiter:str=','):
    assert len(filetuples)
    lengths = [len(tuple) for tuple in filetuples]
    assert all([l == lengths[0] for l in lengths]), 'Unequal tuple lengths'

    lines = []
    for i, tupl in enumerate(filetuples):
        # TODO: relative path
        lines += [', '.join(tupl)]
    txt = '\n'.join(lines)
    open(filepath, 'w').write(txt)



#Helper functions for slicing images (for CHW dimension ordering)
def grid_for_patches(
    imageshape: tp.Tuple[int,int]|torch.Size, 
    patchsize:  int, 
    slack:      int,
) -> np.ndarray:
    assert len(imageshape) == 2
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
    return slice_into_patches_from_grid(image, grid)


def slice_into_patches_from_grid(
    image: torch.Tensor,
    grid:  np.ndarray,
) -> tp.List[torch.Tensor]:
    assert grid.ndim in [2,3] and grid.shape[-1] == 4
    patches = [image[...,i0:i1, j0:j1] for i0,j0,i1,j1 in grid.reshape(-1, 4)]
    return patches


def stitch_overlapping_patches(
    patches:        tp.List[torch.Tensor], 
    imageshape:     tp.Tuple[int,int]|torch.Size, 
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



def normalize(x:torch.Tensor, axis:int, eps:float=1e-5) -> torch.Tensor:
    '''L2 normalization along axis'''
    # TODO: torch.cdist
    return x / (x**2).sum(dim=axis, keepdim=True).clamp_min(eps)**0.5

def l1_normalize(x:torch.Tensor, axis:int, eps:float=1e-5) -> torch.Tensor:
    '''L1 normalization along axis'''
    return x / x.sum(dim=axis, keepdim=True).clamp_min(eps)



def within_bounds_coordinates_mask(
    c_xy:       torch.Tensor, 
    imageshape: tp.Tuple[int,int],
) -> torch.Tensor:
    assert c_xy.ndim == 2 and c_xy.shape[1] == 2
    
    shape = torch.as_tensor(imageshape)
    mask = (
        torch.all(c_xy >= 0, dim=1)
        & (c_xy[:,0] < shape[1])
        & (c_xy[:,1] < shape[0])
    )
    return mask

def filter_out_of_bounds_coordinates(
    c_xy:       torch.Tensor, 
    imageshape: tp.Tuple[int,int],
) -> torch.Tensor:
    mask = within_bounds_coordinates_mask(c_xy, imageshape)
    return c_xy[mask]

def assert_coordinates_within_bounds(c_xy:torch.Tensor, imageshape:tp.Tuple[int,int]):
    assert c_xy.ndim == 2 and c_xy.shape[1] == 2

    assert torch.all(c_xy >= 0)
    shape = torch.as_tensor(imageshape)
    assert torch.all(c_xy[:,0] < shape[1])  # x
    assert torch.all(c_xy[:,1] < shape[0])  # y

def sample_tensor_at_coordinates(
    t:    torch.Tensor, 
    c_xy: torch.Tensor,
    padding_mode:tp.Optional[str] = None, # zeros, border, reflection
) -> torch.Tensor:
    t_ndim = t.ndim
    c_ndim = c_xy.ndim

    assert t_ndim in [3,4], f'Expected [BCHW] or [CHW], got {t.shape}'
    assert c_ndim in [2,3] and c_xy.shape[-1] == 2, \
        f'Expected [BN2] or [N2], got {c_xy.shape}'
    if t_ndim == 3:
        t = t[None]
    if c_ndim == 2:
        c_xy = c_xy[None]
    assert t.shape[0] == c_xy.shape[0]
    
    H,W = t.shape[-2:]
    if padding_mode is None:
        assert_coordinates_within_bounds(c_xy.reshape(-1,2), (H,W))

    # scale to -1..+1
    # NOTE: +0.5 so that x[10,5] == sample_tensor_at_coordinates(x, [5,10])
    c_scaled = (c_xy + 0.5) / torch.as_tensor([W,H], device=c_xy.device) *2 -1
    samples  = torch.nn.functional.grid_sample(
        t, 
        c_scaled[:,None], 
        mode          = 'bilinear', 
        align_corners = False,
        padding_mode  = padding_mode if padding_mode is not None else 'zeros',
    )[:,:,0]

    if t_ndim == 3:
        samples = samples[0]
    return samples


def rot90_coordinates(
    c_xy:       torch.Tensor, 
    imageshape: tp.Tuple[int,int], 
    k:          int,
) -> torch.Tensor:
    assert c_xy.ndim >= 1 and c_xy.shape[-1] == 2
    c_shape = c_xy.shape
    c_xy = c_xy.reshape(-1,2)
    
    k   = k % 4
    dev = c_xy.device
    H,W = torch.tensor(imageshape, device=dev) - 1
    
    if k == 1:
        c_xy = (
            torch.tensor([0,H]).to(dev) 
            + torch.flip(c_xy, dims=[-1]) 
            * torch.tensor([1,-1]).to(dev)
        )
    if k == 2:
        c_xy = (
            torch.tensor([W,H]).to(dev) 
            + c_xy 
            * torch.tensor([-1,-1]).to(dev)
        )
    if k == 3:
        c_xy = (
            torch.tensor([W,0]).to(dev) 
            + torch.flip(c_xy, dims=[-1]) 
            * torch.tensor([-1,1]).to(dev)
        )
    return c_xy.reshape(c_shape)


def adjust_coordinates_for_crop(
    c_xy:     torch.Tensor,
    cropbox:  tp.Tuple[int,int,int,int],
    new_size: tp.Tuple[int,int],
) -> torch.Tensor:
    '''Modify coordinates c_xy as if the corresponding image was cropped at
       coordinates provided by cropbox and scaled to new_size.
       (new_size is width,height, cropbox is left,top,width,height) '''
    assert c_xy.ndim >= 1 and c_xy.shape[-1] == 2

    left, top, width, height = cropbox
    new_width, new_height = new_size

    new_c_xy = torch.stack([
        (c_xy[..., 0] - left) / width * new_width,
        (c_xy[..., 1] - top)  / height * new_height,
    ], dim=-1)
    return new_c_xy




def __unique_and_indices_for_faster_unique(x:torch.Tensor):
    assert x.ndim == 2 and x.shape[-1] == 2
    if len(x) == 0:
        return x, torch.empty([0], dtype=torch.int64, device=x.device)
    
    x1_max = x[:,1].max()
    # hashing values
    y = x[:,0] * (x1_max + 1) + x[:,1]
    _, unique_indices = torch.unique(y, return_inverse=True)
    q = torch.empty(
        [int(unique_indices.max()+1), 2], 
        dtype  = x.dtype, 
        device = x.device,
    )
    q[unique_indices] = x
    return q, unique_indices

@torch.jit.script_if_tracing
def faster_unique_dim0(x:torch.Tensor) -> torch.Tensor:
    '''A faster torch.unique(x, dim=0) for x.shape == [N,2]'''
    q, _ = __unique_and_indices_for_faster_unique(x)
    return q

@torch.jit.script_if_tracing
def faster_unique_dim0_with_counts(x:torch.Tensor) \
    -> tp.Tuple[torch.Tensor, torch.Tensor]:
    '''A faster torch.unique(x, dim=0, return_counts=True) for x.shape == [N,2]'''
    q, indices = __unique_and_indices_for_faster_unique(x)
    counts = torch.bincount(indices, minlength=q.shape[0])
    return q, counts


def _linspace_on_tensors(p0:torch.Tensor, p1:torch.Tensor, n:int) -> torch.Tensor:
    '''torch.linspace() but accepts tensors'''
    assert p0.shape == p1.shape
    assert p0.shape[-1:] == (2,) and p1.shape[-1:] == (2,)

    shape = (n,)+p0.shape
    p0 = p0.reshape(-1,2)
    p1 = p1.reshape(-1,2)
    
    direction = (p1 - p0)
    flat      = p0 + direction * torch.linspace(0,1, n, device=p0.device)[:,None,None]
    return flat.reshape(shape)
