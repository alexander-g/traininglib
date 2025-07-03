from traininglib.paths import pathutils as utils

import torch


def test_path_component_length():
    p = torch.as_tensor([
        (10., 20.),
        (20,  20,),
        (100, 100)
    ])

    plen_fn  = torch.jit.script(utils.path_component_length)
    length   = plen_fn(p)
    assert abs(length - 123.1370849) < 0.00001

    seglengths = utils.path_segment_lengths(p)
    assert len(seglengths) == 2
    assert seglengths[0] == 10
    assert abs(seglengths[1] - 113.1370849) < 0.00001


def test_resample_path_component():
    p = torch.as_tensor([
        (10., 20.),
        (20,  20,),
        (100, 100)
    ])

    p_new = utils.resample_path_component(p, n=10)
    print(p_new)
    assert len(p_new) == 10
    assert torch.all(p_new[1] > 20)
    assert torch.all(p_new[1:,0] == p_new[1:,1])


def test_resample_path_component_at_interval():
    p = torch.as_tensor([]).reshape(-1,2)
    # actual bug
    interval = torch.rand(1) * 10
    #dont fail
    utils.resample_path_component_at_interval(p, interval=interval, last_point_threshold=3)



def test_filter_out_of_bounds_lines():
    lines = torch.as_tensor([
        [[-20,-20],[-10,-10]], # yes filter out
        [[-10,-10],[10,10]],   # no filter out
        [[10,10],[20,20]],
        [[20,20],[25,20]],
    ])

    fn = utils.filter_out_of_bounds_lines
    fn = torch.jit.script(utils.filter_out_of_bounds_lines)
    filtered = fn(lines, size=50)
    assert len(filtered) == 3
    assert torch.all( filtered == lines[1:] )

def test_encode_numpy_paths():
    # actual bug, dont raise error
    encoded = utils.encode_numpy_paths([], 1)
    assert encoded.shape == (0,4)

def test_rasterize_multiple_paths_batched():
    empty_paths = torch.empty([0,4])
    rasterized  = utils.rasterize_multiple_paths_batched(empty_paths, 1, 77)
    # actual bug
    assert rasterized.shape == (1,77,77)


def test_convert_lines_to_pixelcoords():
    size  = 50
    lines = torch.as_tensor([
        [(10,20),(20,60)],
        [(20,60),(90,90)],  # out of bounds
        [(10,10),(20,20)],
    ]).float()
    
    points, indices = utils.convert_lines_to_pixelcoords(lines, size)
    assert len(points) == len(indices)
    assert torch.all( torch.isin(indices, torch.as_tensor([0,2])) )


