import typing as tp

import torch

from . import svg
from ..segmentation.skeletonization import _linspace_on_tensors
from ..datalib import normalize


# width, height
ImageSize = tp.Tuple[int,int]
IntOrSize = tp.Union[int, ImageSize]


DEFAULT_STEPFACTOR:float = 2.0



@torch.jit.script_if_tracing
def rasterize_path_batched(path:torch.Tensor, B:int, size:int) -> torch.Tensor:
    '''Convert a set of points to a binary raster image. One path per image.'''
    assert path.ndim == 2 and path.shape[1] == 3

    # append the batch id as path id
    paths_with_label = torch.cat([path, path[:,-1:]], dim=1)
    return rasterize_multiple_paths_batched(paths_with_label, B, size)

@torch.jit.script_if_tracing
def rasterize_multiple_paths_batched(
    paths:     torch.Tensor, 
    n_batches: int, 
    size:      int,
    stepfactor:float = DEFAULT_STEPFACTOR,
) -> torch.Tensor:
    '''Convert a set of points to a binary raster image. Multiple paths per image.'''
    assert paths.ndim == 2 and paths.shape[1] == 4
    assert n_batches > 0

    rasterized:tp.List[torch.Tensor] = []
    for batch_i in range(n_batches):
        paths_this_image = paths[ paths[:,2] == batch_i ]
        rasterized_i = \
            rasterize_multiple_paths(paths_this_image[:,(0,1,3)], size, stepfactor)

        rasterized += [rasterized_i]
    return torch.stack(rasterized)


def normal_of_line(line:torch.Tensor, unitlength:bool = False) -> torch.Tensor:
    '''Compute the normal of [...,2] vectors.'''
    # NOTE: not [...,2,2] lines
    assert line.shape[-1] == 2

    normal = torch.flip(line, dims=[-1]) * torch.tensor([1,-1], device=line.device)
    if unitlength:
        normal = normalize(normal, axis=-1)
    return normal


@torch.jit.script_if_tracing
def convert_lines_to_pixelcoords(
    lines:  torch.Tensor, 
    size:   int|None,
    offset: float = 0.0,
    round:  bool  = True,
    stepfactor: float = DEFAULT_STEPFACTOR,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    '''Interpolate points on the provided lines (shape [N,2,2]). 
       Returns points (shape [M,2]) and line indices for each point
       - If `size` is not None, will filter out-ouf-bounds points
       - If `offset` is not 0, will shift points orthogonally to the line
       - If `round`, will round points to an integer
    '''
    assert lines.ndim == 3 and lines.shape[1] == 2 and lines.shape[2] == 2, lines.shape
    assert offset == 0, NotImplemented

    lines   = lines.float()
    indices = torch.arange(len(lines), device=lines.device)
        
    if size is not None:
        lines_valid = are_lines_within_bounds(lines, size)
        lines   = lines[lines_valid]
        indices = indices[lines_valid]
    
    p0 = lines[:,0]
    p1 = lines[:,1]
    d  = elementwise_distance(p0, p1, -1)
    n_points = torch.maximum(torch.tensor(2), (d*stepfactor).long())

    j = 0
    all_points_  = [torch.empty_like(lines.reshape(-1,2)[:0])]
    all_indices_ = [torch.empty(0, dtype=torch.long, device=lines.device)]
    # interpolate inbetween the line endpoints (in batches)
    for n in torch.unique(n_points):
        mask = (n_points == n)
        B = mask.sum()

        line_points = _linspace_on_tensors(p0[mask], p1[mask], n)
        if round:
            line_points = line_points.round()
        
        line_indices = torch.ones([n,B], device=lines.device, dtype=torch.long)
        line_indices = line_indices * torch.arange(B)[None] + j
        j += B

        # TODO:
    #     if offset != 0:
    #         linevector  = p1 - p0
    #         normal      = normal_of_line(linevector, unitlength=True)
    #         line_points = line_points + normal * offset

        all_points_.append(line_points.reshape(-1,2))
        all_indices_.append(line_indices.reshape(-1))
    
    all_points  = torch.cat(all_points_)
    all_indices = torch.cat(all_indices_)
    
    if size is not None:
        all_points_valid = points_within_bounds_mask(all_points, size)
        all_points  = all_points[all_points_valid]
        all_indices = all_indices[all_points_valid]
    
    return all_points, all_indices


@torch.jit.script_if_tracing
def rasterize_lines(
    lines: torch.Tensor, 
    size:  IntOrSize, 
    stepfactor:float = 2
) -> torch.Tensor:
    '''Convert a set of lines to a binary raster image. 
       Points in xy ordering, size in wh ordering (or single int).'''
    points_to_paint, _ = \
        convert_lines_to_pixelcoords(lines, size, stepfactor=stepfactor)

    if isinstance(size, int):
        size = (size, size)
    size = size[::-1]
    result = torch.zeros(size, dtype=torch.bool, device=lines.device)
    if len(points_to_paint) == 0:
        # all outside the image
        return result

    # scatter the points onto the result
    points_to_paint = points_to_paint.long()
    result = torch.index_put(
        result, 
        ( points_to_paint[:,1], points_to_paint[:,0] ),
        torch.ones(len(points_to_paint), dtype=result.dtype, device=result.device)
    )
    return result

#TODO: path component
def rasterize_path(path:torch.Tensor, size:int, stepfactor:float=2.0) -> torch.Tensor:
    '''Convert a path to a binary raster image. (xy ordering)'''
    lines = path_to_lines(path)
    return rasterize_lines(lines, size, stepfactor=stepfactor)

def rasterize_multiple_paths(
    paths: torch.Tensor, 
    size:  IntOrSize,
    stepfactor:float = DEFAULT_STEPFACTOR,
) -> torch.Tensor:
    '''Convert a set of paths (encoded as x,y,label) to a binary raster image.'''
    assert paths.ndim == 2 and paths.shape[1] == 3
    
    lines = [torch.empty([0,2,2], dtype=paths.dtype, device=paths.device)]
    for i in torch.unique(paths[:,2]):
        path_i  = paths[ paths[:,2] == i, :2 ]
        lines_i = path_to_lines(path_i)
        lines.append(lines_i)
    return rasterize_lines(torch.cat(lines), size, stepfactor)

def path_to_lines(path:torch.Tensor) -> torch.Tensor:
    '''Convert a [N,2]-path to [N-1,2,2] lines represented by start and end point'''
    # NOTE: removed ndim == 2 assert because not relevant
    #assert path.ndim == 2
    assert path.ndim >= 1
    return torch.stack( [path[:-1], path[1:]], dim=1 )

def paths_to_lines(paths:tp.List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([path_to_lines(p) for p in paths])

def are_lines_within_bounds(lines:torch.Tensor, size:IntOrSize) -> torch.Tensor:
    '''Compute a boolean mask indicating if lines are within a square'''
    assert lines.ndim == 3 and lines.shape[1] == 2 and lines.shape[2] == 2

    linepoints = lines.reshape(-1,2)
    validmask  = points_within_bounds_mask(linepoints, size)
    validmask  = validmask.reshape(-1,2)
    # sanity check for debugging
    assert len(lines) == len(validmask)
    # a line is valid if at least one end point is valid
    validmask  = validmask.any(-1)
    return validmask

# TODO: this function should not filter lines with nodes on opposite borders
def filter_out_of_bounds_lines(lines:torch.Tensor, size:IntOrSize) -> torch.Tensor:
    validmask = are_lines_within_bounds(lines, size)
    return lines[validmask].reshape(-1,2,2)

def points_within_bounds_mask(points:torch.Tensor, size:IntOrSize) -> torch.Tensor:
    assert points.ndim == 2
    if isinstance(size, int):
        size = (size, size)
    return  (
        torch.all( (points >= 0), dim=1 ) 
        & (points[:,0] < size[0])  
        & (points[:,1] < size[1])
    )

def filter_out_of_bounds_points(points:torch.Tensor, size:IntOrSize) -> torch.Tensor:
    assert points.ndim == 2

    valid  = points_within_bounds_mask(points, size)
    return points[valid]

def convert_multiple_paths_tensor_to_svg(paths:torch.Tensor) -> tp.List[svg.Path]:
    '''Convert paths, encoded as [N,3] tensor to a list of list of tuples'''
    assert paths.ndim == 2 and paths.shape[1] == 3
    result:tp.List[svg.Path] = []

    path_ids = torch.unique(paths[:,-1])
    for path_id in path_ids:
        path_i = paths[ paths[:,-1] == path_id, :2 ]
        component = path_i.numpy().tolist()
        component = [tuple(p) for p in component]

        result.append([component])
    return result

def convert_path_component_to_tensor(component:svg.Component) -> torch.Tensor:
    return torch.as_tensor(component)


def elementwise_distance(p0:torch.Tensor, p1:torch.Tensor, dim:int) -> torch.Tensor:
    return (((p0 - p1)**2).sum(dim)**0.5)

def path_segment_lengths(component:torch.Tensor) -> torch.Tensor:
    '''Lengths of individual lines in a path'''
    lines = path_to_lines(component)
    return elementwise_distance(lines[:,0], lines[:,1], dim=-1)

def path_component_length(component:torch.Tensor) -> float:
    '''Lengths of a path'''
    return float(path_segment_lengths(component).sum())

@torch.jit.script_if_tracing
def encode_paths(
    paths:    tp.List[torch.Tensor], 
    batch_nr: int,
    device:   tp.Optional[torch.device] = None,
) -> torch.Tensor:
    '''Convert a list of [N,2] paths tensor into the format [M,4]
       (x, y, batch_nr, path_nr)'''

    device = torch.device('cpu') if device is None else device
    if len(paths) == 0:
        return torch.empty([0,4], device=device)
    
    # in case paths contains numpy arrays
    paths = [torch.as_tensor(p) for p in paths]
    return torch.cat([
        torch.cat(
            [
                torch.as_tensor(p).to(device),
                torch.ones([len(p), 1], dtype=p.dtype, device=device) * batch_nr,
                torch.ones([len(p), 1], dtype=p.dtype, device=device) * path_nr,
            ], 
            dim=1
        ) for path_nr,p in enumerate(paths)
    ] , 
    dim=0
    )

# legacy name
encode_numpy_paths = encode_paths

def decode_paths(paths:torch.Tensor, n_batches:int) -> tp.List[tp.List[torch.Tensor]]:
    '''Convert a tensor as returned by `encode_paths()` to a list of paths'''
    assert paths.ndim == 2 and paths.shape[1] == 4
    assert n_batches > 0

    all_paths_list:tp.List[tp.List[torch.Tensor]] = []
    for batch_i in range(n_batches):
        paths_this_image_list = []
        paths_this_image = paths[ paths[:,2] == batch_i ]
        path_labels      = torch.unique(paths_this_image[:,3])
        
        for label_i in path_labels:
            path = paths_this_image[paths_this_image[:,3] == label_i]
            path = path[:,:2]
            paths_this_image_list.append(path)
        all_paths_list.append(paths_this_image_list)
    return all_paths_list


def resample_path_component_at_segment_lengths(
    component: torch.Tensor, 
    lengths:   torch.Tensor,
) -> torch.Tensor:
    '''Resample a path component so that the lengths of individual segments 
       are as specified in `steps`. Lossy.'''
    assert lengths.ndim == 1 and torch.all(lengths >= 0.0)
    assert component.ndim == 2 and component.shape[1] == 2

    if len(component) == 0:
        return torch.empty_like(component)

    segment_lengths = path_segment_lengths(component)
    t_old = torch.cat([
        torch.zeros(1, device=component.device), 
        torch.cumsum(segment_lengths, 0)
    ])
    t_new = torch.cumsum(lengths, 0)
    #assert torch.abs(t_new[-1] - t_old[-1]) < 1e-3, (t_new[-1], t_old[-1])

    i = 0
    t_now = torch.tensor(0.0)
    p_now = component[0]
    component_new = torch.as_tensor(p_now)[None]
    for t_next in t_new[1:]:
        p_now = component_new[-1]
        while 1:
            t_remainder = t_old[i+1] - t_now
            if t_next - t_old[i+1] < 0.01:
                direction = component[i+1] - component[i]
                direction = direction / segment_lengths[i] # * t_remainder
                p_now     = p_now + direction * (t_next - t_now)
                component_new = torch.cat([component_new, p_now[None]])
                t_now = t_next
                break
            else:
                t_now = t_old[i+1]
                p_now = component[i+1]
                i += 1
    return component_new

def resample_path_component(component:torch.Tensor, n:int) -> torch.Tensor:
    '''Resample a path component so that it contains `n` points. Lossy.'''
    assert component.ndim == 2 and component.shape[1] == 2

    total_length = path_component_length(component)
    t_new   = torch.linspace(0, total_length, n)
    t_delta = torch.cat([torch.zeros(1), torch.diff(t_new)])
    return resample_path_component_at_segment_lengths(component, t_delta)

def resample_path_component_at_interval(
    component: torch.Tensor, 
    interval:  float,
    last_point_threshold: float,
) -> torch.Tensor:
    '''Resample a path component so that the distance between points is `interval`.
       The last point is copied over if it is more than `last_point_threshold`
       away from the n-1 point.'''
    assert component.ndim == 2 and component.shape[1] == 2

    total_length = path_component_length(component)
    t_new = torch.arange(0, total_length, interval).to(component.device)
    if (total_length - t_new[-1]) > last_point_threshold:
        t_new = torch.cat([t_new, torch.as_tensor(total_length)[None].to(t_new.device) ])
    t_delta = torch.cat([torch.zeros(1).to(t_new.device), torch.diff(t_new)])
    return resample_path_component_at_segment_lengths(component, t_delta)


def _randomize_lengths(lengths:torch.Tensor, factor:float) -> torch.Tensor:
    assert lengths.ndim == 1 and torch.all(lengths > 0)
    total = lengths.sum()

    randomized_lengths = lengths * (1 + torch.empty_like(lengths).uniform_(-factor, factor))
    # ensure all values are still positive
    randomized_lengths = torch.clamp(randomized_lengths, min=0.1)
    # scale to have the same sum as the original
    randomized_lengths = randomized_lengths * (total / randomized_lengths.sum())

    return randomized_lengths

def _randomize_n_elements(lengths:torch.Tensor, n_factor) -> torch.Tensor:
    assert lengths.ndim == 1

    n     = len(lengths)
    new_n = n * (1 + (torch.rand(1) * 2 - 1) * n_factor)
    new_n = int(round(float(new_n)))

    while new_n < len(lengths):
        # pick the largest segments and split them
        i = int(lengths.argmin())
        i = max(0, i-1)
        l = lengths[i: i+2]
        lengths = torch.cat([
            lengths[:i],
            l.sum()[None],
            lengths[i+len(l):],
        ])

    while new_n > len(lengths):
        i = int(lengths.argmax())
        l = lengths[i:i+1]
        r = torch.rand(1).uniform_(0.25, 0.75)
        lengths = torch.cat([
            lengths[:i],
            torch.as_tensor([l*r, l*(1-r)]),
            lengths[i+len(l):],
        ])

    return lengths

def randomize_path_component(
    component:            torch.Tensor,
    jitter_factor:        float = 2.0,
    length_jitter_factor: float = 0.3,
    n_segment_factor:     float = 0.2,
) -> torch.Tensor:
    assert component.ndim == 2 and component.shape[1] == 2

    jitter    = torch.rand_like(component) * (jitter_factor*2) - jitter_factor
    component = component + jitter

    lengths = path_segment_lengths(component)
    lengths = _randomize_lengths(lengths, length_jitter_factor)
    lengths = _randomize_n_elements(lengths, n_segment_factor)
    lengths = torch.cat([torch.zeros(1), lengths])
    component = resample_path_component_at_segment_lengths(component, lengths)

    return component


