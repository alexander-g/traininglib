import torch


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
