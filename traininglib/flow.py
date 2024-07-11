import typing as tp

import numpy as np
import torch
import scipy

from . import datalib
from .datalib import ImageSize, TensorPair

InterpMode = tp.Literal['nearest', 'bilinear']

@torch.jit.script_if_tracing
def random_warp(
    x:         torch.Tensor, 
    magnitude: range,
    mode:      InterpMode,
) -> TensorPair:
    '''Randomly warp a tensor. Returns warped tensor and flow'''
    assert len(x.shape) == 4, x.shape
    B,C,H,W = x.shape
    grid    = sparse_grid( (H,W) )
    flows:tp.List[torch.Tensor] = []
    for i in range(len(x)):
        perturbed_grid = perturb_grid(grid, magnitude)
        flow = grids_to_flow(grid, perturbed_grid)
        flows.append(
            torch.as_tensor(flow).float()
        )
    flow_batch = torch.stack(flows).permute(0,3,1,2).to(x.device)
    warped_x = warp_tensor_with_flow(x, flow_batch, mode)
    return warped_x, flow_batch


def warp_tensor_with_flow(
    data: torch.Tensor, 
    flow: torch.Tensor, 
    mode: InterpMode,
) -> torch.Tensor:
    '''Transform a tensor with a given flow field (batched)'''

    assert len(data.shape) == 4
    assert len(flow.shape) == 4
    assert flow.shape[1]   == 2
    assert data.shape[-2:] == flow.shape[-2:]

    h,w  = data.shape[-2:]
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0,w,w), 
            torch.linspace(0,h,h), 
            indexing='xy',
        ),
        dim = -1,
    ).to(flow.device)
    grid = grid[None] - flow.permute(0,2,3,1)
    #scale to -1...+1 (disregarding flow) as required by grid_sample()
    grid[...,0] = grid[...,0] * 2 / w - 1
    grid[...,1] = grid[...,1] * 2 / h - 1
    out  = torch.nn.functional.grid_sample(
        data.to(grid.device), 
        grid, 
        mode, 
        align_corners=True,
    )
    return out

def warp_coordinates_with_flow(xy:torch.Tensor, flow:torch.Tensor) -> torch.Tensor:
    assert xy.ndim == 3 and xy.shape[-1] == 2, xy.shape
    assert len(flow.shape) == 4
    assert flow.shape[1]   == 2

    flow_at_coords = datalib.sample_tensor_at_coordinates(flow, xy)
    flow_at_coords = flow_at_coords.permute(0,2,1)
    new_coords     = xy + flow_at_coords
    return new_coords


def sparse_grid(size:ImageSize, n:tp.Tuple[int,int] = (5,5)) -> np.ndarray:
    '''Generate `n` equally spaced coordinates for image dimensions `size`'''
    W,H = size
    return np.stack(
        np.meshgrid(
            np.linspace(0, W, n[0]),
            np.linspace(0, H, n[1]),
            indexing='xy',
        ), 
        axis=-1
    )

def perturb_grid(grid:np.ndarray, magnitude_range:range) -> np.ndarray:
    '''Randomize a grid for warping'''
    magnitude = np.random.uniform(magnitude_range.start, magnitude_range.stop)
    noise     = np.random.uniform(-1, 1, size=grid.shape) * magnitude
    
    #clip to edges
    noise[0, :,1] = 0
    noise[-1,:,1] = 0
    noise[:, 0,0] = 0
    noise[:,-1,0] = 0
    return grid + noise

def grids_to_flow(grid0:np.ndarray, grid1:np.ndarray) -> np.ndarray:
    '''Generate a flow field from `grid0` to `grid1` of shape (H,W,2)'''
    interp = scipy.interpolate.LinearNDInterpolator(grid1.reshape(-1,2), grid0.reshape(-1,2))
    W,H = grid0[-1,-1]
    X,Y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    return interp(X,Y) - np.stack([X,Y], axis=-1)

