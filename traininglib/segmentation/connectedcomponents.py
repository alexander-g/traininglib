import torch
from .segmentationmodel import connected_components_max_pool


def connected_components_from_adjacency_list(adjlist:torch.Tensor) -> torch.Tensor:
    '''Label connected components in a graph, provided in form of an 
       adjacency list, via depth-first search.'''
    assert adjlist.ndim == 2
    assert adjlist.shape[0] > 0
    assert adjlist.shape[1] == 2

    relabeled = adjlist.clone()
    visited   = torch.zeros(adjlist.shape, dtype=torch.bool)

    cnt = 0
    while not torch.all(visited):
        stack = torch.argwhere(~visited)[:1]
        while len(stack):
            index   = stack[-1]
            i,j     = index[0], index[1]
            not_j   = 1-j
            label_i = relabeled[i, j]
            label_j = relabeled[i, not_j]

            relabeled[i, :] = label_i
            visited[i, :]   = 1
            mask            = (relabeled==label_j) & ~visited
            relabeled[mask] = label_i
            visited[mask]   = 1

            stack = torch.cat([stack[:-1], torch.argwhere(mask)])
    return relabeled


@torch.jit.script_if_tracing
def _relabel(
    labelmap:         torch.Tensor, 
    adjacency_list:   torch.Tensor, 
    adjacency_labels: torch.Tensor
) -> torch.Tensor:
    new_labelmap = labelmap.clone()
    uniques      = torch.unique(adjacency_labels)
    for i in uniques:
        equivalent_labels = adjacency_list[(adjacency_labels == i)]
        mask = torch.isin(labelmap, equivalent_labels)
        new_labelmap[mask] = i
    return new_labelmap


def connected_components_patchwise(x:torch.Tensor, patchsize:int) -> torch.Tensor:
    assert x.ndim == 4
    H,W = x.shape[-2:]
    y   = torch.zeros(x.shape, dtype=torch.int64)
    nel = 0
    for i in range(0, H, patchsize):
        for j in range(0, W, patchsize):
            xpatch = x[..., i:, j:][...,:patchsize, :patchsize]
            ypatch = connected_components_max_pool(xpatch, start=nel)
            y[...,i:, j:][..., :patchsize, :patchsize] = ypatch
            nel += ypatch.numel()
    
    adjacency_list = torch.empty([0,2], dtype=torch.int64)
    #perform connected components at the borders #TODO: code re-use
    for i in range(patchsize-1, H, patchsize):
        yborder = y[..., i:, :][..., :2, :]
        bordercomponents = connected_components_max_pool(yborder.bool())
        for l in torch.unique(bordercomponents):
            if l == 0:
                continue
            eq_i = torch.unique( yborder[ (bordercomponents == l) ] )
            adjacency_list = torch.cat([
                adjacency_list, 
                torch.stack([eq_i[:-1], eq_i[1:]], dim=1)
            ])

    for j in range(patchsize-1, W, patchsize):
        yborder = y[..., :, j:][..., :, :2]
        bordercomponents = connected_components_max_pool(yborder.bool())
        for l in torch.unique(bordercomponents):
            if l == 0:
                continue
            eq_j = torch.unique( yborder[ (bordercomponents == l) ] )
            adjacency_list = torch.cat([
                adjacency_list, 
                torch.stack([eq_j[:-1], eq_j[1:]], dim=1)
            ])
    adjacency_labels = connected_components_from_adjacency_list(adjacency_list)
    return _relabel(y, adjacency_list, adjacency_labels)
        



