import typing as tp

import torch


@torch.jit.script_if_tracing
def randperm(n:int) -> torch.Tensor:
    '''torch.randperm(n) but ONNX-exportable'''
    return torch.argsort(torch.rand(n))

@torch.jit.script_if_tracing
def pad_to_n(x:torch.Tensor, n:int) -> torch.Tensor:
    xsize = torch.as_tensor(x.size())
    H     = xsize[2]
    W     = xsize[3]
    pad_y = int( (torch.ceil(H / n).long() * n) - H )
    pad_x = int( (torch.ceil(W / n).long() * n) - W )
    x_padded = torch.nn.functional.pad(x, [0, pad_x, 0, pad_y])
    return x_padded

@torch.jit.script_if_tracing
def maxpool_3x3_strided(x:torch.Tensor, stride:int) -> torch.Tensor:
    x_padded = pad_to_n(x, n=stride)
    x_pooled = torch.nn.functional.max_pool2d(
        x_padded, kernel_size=3, stride=stride, padding=0
    )
    x_padded_size = x_padded.size()
    x_resized = torch.nn.functional.interpolate(
        x_pooled, [x_padded_size[2], x_padded_size[3]], mode='nearest'
    )
    # ones_3x3 = torch.ones(1,1,3,3)
    # x_resized = torch.nn.functional.conv_transpose2d(x_pooled, ones_3x3, stride=3)
    xsize    = x.size()
    x_sliced = x_resized[..., :xsize[2], :xsize[3]]
    return x_sliced

@torch.jit.script_if_tracing
def connected_components_max_pool(x:torch.Tensor, start:int = 0) -> torch.Tensor:
    '''Inefficient connected components algorithm via maxpooling'''
    assert x.dtype == torch.bool
    assert x.ndim == 4

    x = x.byte()
    n = len(x.reshape(-1))
    labeled = torch.arange(start,start+n, dtype=torch.float32, device=x.device)
    #shuffling makes convergence faster (in torchscript but not in onnx)
    #labeled = labeled[randperm(n)]
    labeled = labeled.reshape(x.size()) * x

    i = 0
    while 1:
        update = torch.nn.functional.max_pool2d(
            labeled, kernel_size=3, stride=1, padding=1
        )
        
        update = update * x
        change = (update != labeled)
        if not torch.any(change):
            break
        labeled = update
        i += 1
    return labeled.long()

@torch.jit.script_if_tracing
def connected_components_transitive_closure(adjmat:torch.Tensor) -> torch.Tensor:
    '''Connected components on an adjacency matrix via mm multiplication'''
    assert adjmat.dtype == torch.bool and adjmat.ndim == 2
    n = adjmat.shape[0]

    # iterative transitive closure computation
    # start with direct reachability (including self)
    R = adjmat | torch.eye(n, dtype=torch.bool, device=adjmat.device)
    prev:tp.Optional[torch.Tensor] = None
    while prev is None or not torch.equal(prev, R):
        prev = R #.clone()
        R_f32 = R.to(torch.float32)
        R = R | ((R_f32 @ R_f32) > 0)

    labels = -torch.ones(n, dtype=torch.int64, device=adjmat.device)
    label  = 0
    for i in range(n):
        if labels[i] == -1:
            # R contains fully connected components
            ixs = torch.argwhere(R[i])
            labels[ixs] = label
            label += 1
    return labels


def adjacency_list_to_matrix(adjlist:torch.Tensor, n:int) -> torch.Tensor:
    assert adjlist.ndim == 2
    assert adjlist.shape[1] == 2

    adjmat = torch.zeros([n,n], dtype=torch.bool, device=adjlist.device)
    adjmat[adjlist[:,0], adjlist[:,1]] = True
    return adjmat



@torch.jit.script_if_tracing
def dfs_on_adjacency_list(adjlist:torch.Tensor) -> tp.Dict[str,torch.Tensor]:
    '''Label connected components in a graph, provided in form of an 
       adjacency list, via depth-first search.'''
    assert adjlist.ndim == 2
    #assert adjlist.shape[0] > 0
    assert adjlist.shape[1] == 2

    relabeled = adjlist.clone()
    visited   = torch.zeros(adjlist.shape, dtype=torch.bool, device=adjlist.device)

    cnt = 0
    while not torch.all(visited):
        stack = torch.nonzero(~visited)[:1]
        # points in visited order
        verts = stack.clone()
        # the predecessor index for each point in `verts`
        prev  = torch.ones(len(stack), dtype=torch.int64) * -1

        while len(stack):
            index   = stack[-1]
            i,j     = index[0], index[1]
            not_j   = 1-j
            # NOTE: without clone only a view, would be overwritten later
            label_i = relabeled[i, j].clone()
            label_j = relabeled[i, not_j].clone()

            relabeled[i, :] = label_i.repeat(2)
            visited[i, :]   = torch.tensor([True,True])
            mask            = ((relabeled==label_j) | (relabeled==label_i))
            mask            = mask & ~visited
            #relabeled[mask] = label_i # onnx error
            relabeled       = torch.where(mask, label_i, relabeled)
            #visited[mask]   = 1 # onnx error
            visited         = visited | mask

            next  = torch.nonzero(mask)
            stack = torch.cat([stack[:-1], next])
            #prev  = torch.cat([prev, ... )])
            verts = torch.cat([verts, index[None]])
    return {
        'connectedcomponents': relabeled,
    }

@torch.jit.script_if_tracing
def connected_components_from_adjacency_list(adjlist:torch.Tensor) -> torch.Tensor:
    return dfs_on_adjacency_list(adjlist)['connectedcomponents']


@torch.jit.script_if_tracing
def _relabel(
    labelmap:         torch.Tensor, 
    adjacency_list:   torch.Tensor, 
    adjacency_labels: torch.Tensor
) -> torch.Tensor:
    new_labelmap = labelmap.clone()
    uniques      = torch.unique(adjacency_labels)
    for i in uniques:
        equivalent_labels = adjacency_list[(adjacency_labels == i)].reshape(-1)
        # NOTE: torch.isin() is not exportable to onnx
        # mask = torch.isin(labelmap, equivalent_labels)
        for l in equivalent_labels:
            mask = (labelmap == l)
            new_labelmap = torch.where(mask, i, new_labelmap)
    return new_labelmap

@torch.jit.script_if_tracing
def _adjacency_at_borders(x:torch.Tensor, patchsize:int) -> torch.Tensor:
    assert x.ndim == 4
    W       = x.shape[-1]
    adjlist = torch.empty([0, 2], dtype=torch.int64, device=x.device)
    for i in range(patchsize-1, W, patchsize):
        adjlist_i = torch.empty([0,2], dtype=torch.int64, device=x.device)

        xborder   = x[..., i:][..., :2]
        bordercomponents = connected_components_max_pool(xborder.to(torch.bool))
        for l in torch.unique(bordercomponents):
            if l == 0:
                continue
            eq_i      = torch.unique( xborder[bordercomponents==l] )
            adjlist_i = torch.cat([
                adjlist_i, torch.stack([eq_i[:-1], eq_i[1:]], dim=1)
            ])
        
        # dual torch.cat because onnx doesnt like catting in dual-loops
        adjlist = torch.cat([adjlist, adjlist_i])
    return adjlist


@torch.jit.script_if_tracing
def connected_components_patchwise(x:torch.Tensor, patchsize:int) -> torch.Tensor:
    assert x.ndim == 4
    H,W = x.shape[-2:]
    y   = torch.zeros(x.shape, dtype=torch.int64, device=x.device)
    nel = 0
    for i in range(0, H, patchsize):
        for j in range(0, W, patchsize):
            xpatch = x[..., i:, j:][..., :patchsize, :patchsize]
            ypatch = connected_components_max_pool(xpatch, start=nel)
            # NOTE: onnx doesnt like this
            #y[...,i:, j:][..., :patchsize, :patchsize] = ypatch.to(y.dtype)
            y[..., i:i+patchsize, j:j+patchsize] = ypatch.to(y.dtype)
            nel += ypatch.numel()

    adjacency_list = torch.cat([
         _adjacency_at_borders(y.transpose(-2,-1), patchsize),
         _adjacency_at_borders(y, patchsize)
    ])

    adjacency_labels = connected_components_from_adjacency_list(adjacency_list)
    return _relabel(y, adjacency_list, adjacency_labels)


@torch.jit.script_if_tracing
def filter_components(x:torch.Tensor, pixel_threshold:int) -> torch.Tensor:
    assert x.dtype == torch.int64

    x = x.clone()
    labels, counts = torch.unique(x, return_counts=True)
    mask = (counts < pixel_threshold)
    for l in labels[mask]:
        x *= (x != l).to(x.dtype)
    return x

