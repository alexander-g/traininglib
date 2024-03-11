import typing as tp

import torch



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
        return torch.empty([0,2])
    
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
        if v[p[0], p[1]] == 1:
            continue
        distance = dist[p] + 1

        x_nhood  = x[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
        v_nhood  = v[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
        v_nhood[1,1] = 1

        prvhood  = prev[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
        dsthood  = dist[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
        new_mask = (x_nhood & ~v_nhood)
        prvhood[new_mask] = p  #TODO: should not get overwritten (?)
        dsthood[new_mask] = (dist[p[0],p[1]] + 1) #TODO: should not get overwritten

        new_pts  = new_mask.nonzero() + p -1  # -1 because of nhood offset
        stack    = torch.cat([new_pts, stack])

    path = torch.empty([0,2])
    p = get_maximum_distance_points(endpoints)[1] #TODO: use dist.argmax()
    while not torch.any(p == -1):
        path = torch.cat([path, p[None]])
        p = prev[p[0], p[1]]
    #remove the padding and reverse the direction
    path = torch.flip(path - 1, dims=[0])
    return path
