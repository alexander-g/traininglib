import typing as tp

import torch
from ..import datalib


def _compute_neighbors(x:torch.Tensor) -> torch.Tensor:
    '''Compute the values P2...P9 for each pixel via convolutions'''
    kernels = torch.nn.functional.one_hot(torch.as_tensor([1,2,5,8,7,6,3,0]), 9)
    kernels = kernels.reshape([8,1,3,3]).float()
    return torch.nn.functional.conv2d(x.float(), kernels, padding=1).to(torch.bool)

def compute_neighbors(x:torch.Tensor) -> torch.Tensor:
    '''Compute the values P2...P9 for each pixel via padding and slicing'''
    assert x.ndim == 4
    assert x.shape[1] == 1

    x_padded = torch.nn.functional.pad(x, (1,1,1,1))
    P2 = x_padded[...,  :-2, 1:-1]
    P3 = x_padded[...,  :-2, 2:  ]
    P4 = x_padded[..., 1:-1, 2:  ]
    P5 = x_padded[..., 2:  , 2:  ]
    P6 = x_padded[..., 2:  , 1:-1]
    P7 = x_padded[..., 2:  ,  :-2]
    P8 = x_padded[..., 1:-1,  :-2]
    P9 = x_padded[...,  :-2,  :-2]

    return torch.cat([P2,P3,P4,P5,P6,P7,P8,P9], dim=1)

def condition_a(P:torch.Tensor) -> torch.Tensor:
    '''Compute if the number of neighbors of a pixel is >=2 and <=6 '''
    B = P.sum(1)[:,None]
    return (2 <= B) & (B <= 6)

def condition_b(P:torch.Tensor) -> torch.Tensor:
    '''Compute the number of 0-1 transitions in P2...P9'''
    T_23 = ((P[:,0] == 0) & (P[:,1] == 1)).to(torch.int32) # i32 because of onnx
    T_34 = ((P[:,1] == 0) & (P[:,2] == 1)).byte()
    T_45 = ((P[:,2] == 0) & (P[:,3] == 1)).byte()
    T_56 = ((P[:,3] == 0) & (P[:,4] == 1)).byte()
    T_67 = ((P[:,4] == 0) & (P[:,5] == 1)).byte()
    T_78 = ((P[:,5] == 0) & (P[:,6] == 1)).byte()
    T_89 = ((P[:,6] == 0) & (P[:,7] == 1)).byte()
    T_92 = ((P[:,7] == 0) & (P[:,0] == 1)).byte()  # not in paper?
    T_sum = T_23 + T_34 + T_45 + T_56 + T_67 + T_78 + T_89 + T_92
    return (T_sum == 1)[:,None]

def condition_c0(P:torch.Tensor) -> torch.Tensor:
    '''Condition c): P2 * P4 * P6 = 0'''
    return (P[:,0] * P[:,2] * P[:,4] == 0)[:,None]

def condition_d0(P:torch.Tensor) -> torch.Tensor:
    '''Condition d): P4 * P6 * P8 == 0'''
    return (P[:,2] * P[:,4] * P[:,6] == 0)[:,None]

def condition_c1(P:torch.Tensor) -> torch.Tensor:
    '''Condition c): P2 * P4 * P8 = 0'''
    return (P[:,0] * P[:,2] * P[:,6] == 0)[:,None]

def condition_d1(P:torch.Tensor) -> torch.Tensor:
    '''Condition d): P2 * P6 * P8 == 0'''
    return (P[:,0] * P[:,4] * P[:,6] == 0)[:,None]


@torch.jit.script_if_tracing
def skeletonize(x:torch.Tensor) -> torch.Tensor:
    '''Skeletonization as in https://doi.org/10.1145/357994.358023'''
    x = x.to(torch.bool)
    i = 0
    c = torch.tensor(True)
    while c:
        # subiteration 1
        P      = compute_neighbors(x)
        mask   = x & condition_a(P) & condition_b(P)
        mask_0 = mask & condition_c0(P) & condition_d0(P)
        x      = x & (~mask_0)

        # subiteration 2
        P      = compute_neighbors(x)
        mask   = x & condition_a(P) & condition_b(P)
        mask_1 = mask & condition_c1(P) & condition_d1(P)
        x      = x & (~mask_1)

        i += 1
        c = torch.any( mask_0 | mask_1 )
    return x


@torch.jit.script_if_tracing
def skeletonize_indexed(x:torch.Tensor) -> torch.Tensor:
    '''Skeletonization as in https://doi.org/10.1145/35799'''
    assert x.ndim == 2 and x.dtype == torch.bool
    x = torch.nn.functional.pad(x, (1,1, 1,1))

    # [98,2]
    offsets = torch.tensor([
        #[ 0, 0],
        [-1, 0],  # P2
        [-1, 1],  # P3
        [ 0, 1],  # P4
        [ 1, 1],  # P5
        [ 1, 0],  # P6
        [ 1,-1],  # P7
        [ 0,-1],  # P8
        [-1,-1],  # P9
    ], device=x.device)

    condition = True
    while condition:
        # subiteration 1
        indices = torch.argwhere(x)
        # [N,8,2]
        offset_indices = indices[:,None,:] + offsets[None,:,:]
        # [8,9]
        P = x[offset_indices[...,0], offset_indices[...,1]]
        mask_0 = condition_a(P) & condition_b(P) & condition_c0(P) & condition_d0(P)

        bad_indices_0 = indices[mask_0[:,0]]
        x[bad_indices_0[...,0], bad_indices_0[...,1]] = 0



        # subiteration 2
        indices = torch.argwhere(x)
        # [N,8,2]
        offset_indices = indices[:,None,:] + offsets[None,:,:]
        # [N,8]
        P = x[offset_indices[...,0], offset_indices[...,1]]
        mask_1 = condition_a(P) & condition_b(P) & condition_c1(P) & condition_d1(P)

        bad_indices_1 = indices[mask_1[:,0]]
        x[bad_indices_1[...,0], bad_indices_1[...,1]] = 0



        condition = len(bad_indices_0) > 0 or len(bad_indices_1) > 0

    # remove padding
    x = x[1:-1, 1:-1]

    return x




@torch.jit.script_if_tracing
def find_endpoints(x:torch.Tensor) -> torch.Tensor:
    '''Find endpoints of a skeletonized structure'''
    assert x.ndim == 2

    k = torch.ones([1,1,3,3])
    k[..., 1,1] = 0
    r = torch.nn.functional.conv2d(x.float()[None,None], k, padding=1)[0,0]
    return torch.nonzero((r == 1) & x.to(torch.bool))

@torch.jit.script_if_tracing
def distance_matrix(p:torch.Tensor) -> torch.Tensor:
    assert p.ndim == 2
    return ((p[:,None] - p[None])**2).sum(-1)**0.5

@torch.jit.script_if_tracing
def get_maximum_distance_points(p:torch.Tensor) -> torch.Tensor:
    assert p.ndim == 2
    dmat = distance_matrix(p)
    ixs  = torch.nonzero(dmat == dmat.max())[0]
    return torch.stack([
        p[ixs[0]], 
        p[ixs[1]],
    ])

@torch.jit.script_if_tracing
def path_via_dfs(x:torch.Tensor) -> torch.Tensor:
    '''Find the longest path in a skeleton binary image'''
    assert x.ndim == 2
    assert x.dtype == torch.bool

    #pad to prevent negative indices
    x = torch.nn.functional.pad(x, (1,0,1,0))

    endpoints = find_endpoints(x)
    if len(endpoints) < 2:
        return torch.empty([0,2], dtype=torch.int64)
    
    #TODO: error handling if len(endpoints) < 2
    dmat      = distance_matrix(endpoints)
    p0        = get_maximum_distance_points(endpoints)[0]
    stack     = p0[None]
    visited   = v = ~x

    # predecessor nodes per pixel
    prev = -torch.ones(x.shape+(2,)).long()
    # distance to starting point per pixel
    dist = -torch.ones(x.shape).long()
    dist[p0[0], p0[1]] = 0

    while len(stack) > 0:
        p, stack = stack[0], stack[1:]
        py,px    = p[0], p[1]
        if v[p[0], p[1]] == 1:
            new_pts = torch.empty([0,2], dtype=stack.dtype)
            # NOTE: no continue because onnx doesnt like it
            #continue
        else:
            distance = dist[py, px] + 1
            # 3x3 neighborhood
            y0,y1 = py-1, py+2
            x0,x1 = px-1, px+2

            #mark point as visited
            v[py,px] = True

            x_nhood  = x[y0:y1, x0:x1]
            v_nhood  = v[y0:y1, x0:x1]
            # NOTE: onnx error
            #v_nhood[1,1] = 1 
            # NOTE: doesnt do as intented in onnx, (not a view of v)
            #v_nhood[torch.tensor(1), torch.tensor(1)] = True

            #mask indicating the next coordinates to visit
            new_mask = (x_nhood & ~v_nhood)
            
            # NOTE: this doesnt work on onnx for same reason as above
            #prvhood  = prev[y0:y1, x0:x1]
            #dsthood  = dist[y0:y1, x0:x1]
            #prvhood[new_mask] = p
            #dsthood[new_mask] = (dist[p[0],p[1]] + 1)

            # the next coordinates to visit
            new_pts_ = torch.nonzero(new_mask) + p -1  # -1 because of nhood offset
            # NOTE: ugly loop but could not get it to work in onnx without
            for c in new_pts_:
                cy,cx = p[0], p[1]
                #update predecessors
                prev[c[0], c[1], 0] = cy
                prev[c[0], c[1], 1] = cx
                #update distances
                dist[c[0], c[1]]    = dist[cy,cx] + 1
            
            # NOTE: looks like a noop but is required for onnx
            # I believe otherwise it optimizes away the prev and dist updates above
            new_pts = new_pts_
        stack = torch.cat([new_pts, stack])

    #trace the path from endpoint back to starting point
    path = torch.empty([0,2], dtype=torch.int64)
    q = get_maximum_distance_points(endpoints)[1] #TODO: use dist.argmax()
    while not torch.any(q == -1):
        path = torch.cat([path, q[None]])
        q = prev[q[0], q[1]]
        # why onnx, why is q.shape == (1,2) ??? squeezing to remove
        q = torch.squeeze(q)
    #remove the padding and reverse the direction
    path = torch.flip(path - 1, dims=[0])
    return path


@torch.jit.script_if_tracing
def slice_mask(mask:torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    '''Slice mask to remove empty background. 
       Returns the slice and top-left coordinates'''
    assert mask.dtype == torch.bool
    assert mask.ndim == 2

    coordinates     = torch.nonzero(mask)
    if len(coordinates) == 0:
        return (
            torch.empty([0,0], dtype=mask.dtype), 
            torch.zeros([2], dtype=torch.int64)
        )
    top_left, _     = coordinates.min(0)
    bottom_right, _ = coordinates.max(0)
    sliced_mask     = mask[
        top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1
    ]
    return sliced_mask, top_left



@torch.jit.script_if_tracing
def paths_from_labeled_skeleton(
    labeled_skeletonmap:    torch.Tensor,
) -> torch.Tensor:
    '''Convert all skeletons to paths. Returns a Nx3 tensor: (x,y,label) '''
    assert labeled_skeletonmap.dtype == torch.int64

    paths = torch.zeros([0,3], dtype=torch.int64)
    unique_labels = torch.unique(labeled_skeletonmap)
    #remove label zero
    unique_labels = unique_labels[unique_labels != 0]
    for l in unique_labels:
        # NOTE: this `if` would create two sub-graphs 
        # and onnx doesnt like torch.cat in nested graphs
        # ..., so dont
        #if l == 0:
        #    continue

        skeleton, top_left = slice_mask(labeled_skeletonmap == l)
        path = path_via_dfs(skeleton)
        # re-add offset to coordinates
        path = path + top_left
        labeled_path = torch.nn.functional.pad(path, [0,1], value=l)

        paths = torch.cat([paths, labeled_path])
    return paths


# legacy
_linspace_on_tensors = datalib._linspace_on_tensors



@torch.jit.script_if_tracing
def rdp(path:torch.Tensor, epsilon:float) -> torch.Tensor:
    '''Ramer-Douglas-Peucker polyline simplification algorithm.
       torch.script() compatible.
       `path` must be sorted from start of polyline to the end.'''
    assert path.ndim == 2 and path.shape[1] == 2
    assert epsilon > 0.0

    n = len(path)
    if n < 2:
        return path
    
    points_to_keep = torch.zeros(n, dtype=torch.bool, device=path.device)
    stack = torch.as_tensor([0, n-1], device=path.device).reshape(1,2)

    while len(stack) > 0:
        ij    = stack[0]
        i,j   = ij[0], ij[1]
        stack = stack[1:]
        points_to_keep[i] = True
        points_to_keep[j] = True

        segment = path[i:j+1]
        n  = len(segment)
        p0 = segment[0]
        p1 = segment[-1]

        #redneck point-to-line distance
        # TODO redo this properly
        #linepoints = torch.linspace(p0, p1, n)
        linepoints = _linspace_on_tensors(p0, p1, n)
        # TODO: torch.cdist
        distances  = ((segment - linepoints)**2).sum(-1)**0.5
        
        k = distances.argmax()
        if distances[k] > epsilon:
            segment0 = torch.stack([i, k+i])
            segment1 = torch.stack([k+i, j])
            stack = torch.cat([stack, segment0[None], segment1[None]])
    return path[points_to_keep]
