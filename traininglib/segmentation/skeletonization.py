import typing as tp

import torch



def _compute_neighbors(x:torch.Tensor) -> torch.Tensor:
    '''Compute the values P2...P9 for each pixel via convolutions'''
    kernels = torch.nn.functional.one_hot(torch.as_tensor([1,2,5,8,7,6,3,0]), 9)
    kernels = kernels.reshape([8,1,3,3]).float()
    return torch.nn.functional.conv2d(x, kernels, padding=1)

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
    T_23 = ((P[:,0] == 0) & (P[:,1] == 1)).byte()
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



