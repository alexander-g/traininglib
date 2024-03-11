import warnings; warnings.simplefilter('ignore')
import typing as tp

import torch


A = torch.as_tensor([
    [0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,1,0,0],
    [0,0,0,0,1,0,0],
])


def path_via_dfs(x:torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    assert x.dtype == torch.bool

    #TODO: pad to prevent negative indices

    #endpoints = TODO
    endpoints = torch.tensor([ [1,1], [6,5] ])
    stack   = endpoints
    visited = v = ~x

    path = torch.empty([0,2])
    prev = -torch.ones(x.shape+(2,)).long()

    while len(stack) > 0:
        p, stack = stack[0], stack[1:]
        if v[p[0], p[1]] == 1:
            continue
        path = torch.cat([path, p[None]])

        x_nhood  = x[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
        v_nhood  = v[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
        v_nhood[1,1] = 1

        prvhood  = prev[p[0]-1:p[0]+2, p[1]-1:p[1]+2]
        new_mask = (x_nhood & ~v_nhood)
        prvhood[new_mask] = p  #TODO: should not get overwritten

        new_pts  = new_mask.nonzero() + p -1  # -1 because offset
        print('p:', p, 'new:', new_pts, 'stack:', stack, 'v:', v.float().mean())
        print('---------')
        stack    = torch.cat([new_pts, stack])
        if len(stack) > 12:
            break
    print(prev)
    return path

dfs = torch.jit.script(path_via_dfs)
v = dfs(A.bool())
print(v.long())

print('done')

