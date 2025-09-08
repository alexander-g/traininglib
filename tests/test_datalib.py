from traininglib import datalib
import tempfile, os
import numpy as np
import torch

import pytest


def test_load_file_pairs():
    tmpdir = tempfile.TemporaryDirectory()
    filepath = os.path.join(tmpdir.name, 'pairs.csv')
    open(filepath, 'w').write(
        'invalid'
    )

    with pytest.raises(Exception):
        output = datalib.load_file_pairs(filepath)

    open(filepath, 'w').write(
        'nonexistent0.jpg, nonexistent0.png\n'
        'nonexistent1.jpg, nonexistent1.png\n'
    )
    with pytest.raises(Exception):
        output = datalib.load_file_pairs(filepath)


    subfolder = os.path.join('folder', 'subfolder')
    subfolder_abs = os.path.join(tmpdir.name, subfolder)
    os.makedirs(subfolder_abs)
    n = 3
    for i in range(n):
        open(f'{subfolder_abs}/image{i}.jpg', 'w').write('.')
        open(f'{subfolder_abs}/mask{i}.png', 'w').write('.')
    
    open(filepath, 'w').write(
        f'{subfolder}/image0.jpg, {subfolder}/mask0.png\n'
        f'{subfolder}/image1.jpg, {subfolder}/mask1.png\n'
        f'{subfolder}/image2.jpg, {subfolder}/mask2.png\n'
        f'\n'
    )
    output = datalib.load_file_pairs(filepath)
    assert len(output) == 3
    #NOTE: awkward use of join and / because of linux/windows issues
    assert output[1][1] == f'{os.path.join(tmpdir.name, subfolder)}/mask1.png'


def test_slice_stitch_images():
    x         = np.random.random([3,1024,2048])
    slack     = np.random.randint(20,50)
    patchsize = np.random.randint(100,400)
    patches   = datalib.slice_into_patches_with_overlap(x, patchsize, slack)
    
    assert patches[0].shape == (3, patchsize, patchsize)

    y         = datalib.stitch_overlapping_patches(patches, x.shape, slack=slack)

    assert x.shape == y.shape
    assert np.all( x == y.numpy() )


def test_rotate_flip():
    x,t = torch.zeros([1,3,100,100]), torch.zeros([1,3,100,100])
    x[0,2,30,30] = 1
    t[0,2,30,30] = 1
    
    x1,t1 = datalib.random_rotate_flip(x, t)

    assert torch.all( x1 == t1 )
    assert x[:,:2].sum() == 0
    assert (x[0,2,30,30]==1 + x[0,2,30,70]==1 + x[0,2,70,70]==1 + x[0,2,70,30]==1) == 1

def test_random_crop():
    x,t = torch.zeros([1,3,100,100]), torch.zeros([1,3,100,100])
    t[::2,::2] = 7.77

    x1,t1 = datalib.random_crop(x, t, patchsize=77, modes=['bilinear', 'nearest'])

    assert x1.shape == t1.shape == (1,3,77,77)

    #nearest interpolation
    t1_uniques = np.unique(t1)
    assert t1_uniques.shape == (2,)
    assert 0 in t1_uniques
    assert 7.77 in t1_uniques


def test_pad_to_minimum_size():
    x  = torch.ones(np.random.randint(4,50, size=[4]).tolist())
    x2 = datalib.pad_to_minimum_size(x, 65)
    assert x.shape[:2] == x2.shape[:2]
    assert x2.shape[-2] == 65
    assert x2.shape[-1] == 65

    #dont pad if already large enough
    x3 = datalib.pad_to_minimum_size(x2, 10)
    assert x3.shape == x2.shape


import traininglib.segmentation.segmentationmodel as segm

def test_torchscript_paste_patch():
    x         = torch.rand([1,3,1024,2048])
    slack     = torch.randint(20,50, (1,))
    patchsize = torch.randint(100,400, (1,))
    patches   = []
    grid      = segm.grid_for_patches(segm.image_size(x), patchsize, slack)
    n         = len(grid.reshape(-1,4))
    for i in range(n):
        patches += [segm.get_patch_from_grid(x, grid, torch.as_tensor(i))]
    
    assert len(patches) == grid.shape[0]*grid.shape[1]
    assert patches[0].shape == (1, 3, patchsize, patchsize)

    y = torch.zeros_like(x)
    for i, patch in enumerate(patches):
        y = segm.paste_patch(y, patch, grid, torch.as_tensor(i), slack)

    assert np.all( x.numpy() == y.numpy() )


def test_ts_slice_paste_bug0():
    scale = 1/3
    h,w = int(1360*scale), int(1024*scale)
    x = torch.rand([1, 3, h, w])
    patchsize = 512
    slack = 32

    grid = segm.grid_for_patches(segm.image_size(x), patchsize, slack)
    print(grid.shape)

    full_output = torch.ones([1,3,h,w])
    for i in range(len(grid.reshape(-1,4))):
        patch = segm.get_patch_from_grid(x, grid, torch.tensor(i))
        segm.paste_patch(full_output, patch, grid, torch.tensor(i), slack)
    assert (x == full_output).all()


    patches  = datalib.slice_into_patches_with_overlap(x, patchsize, slack)
    stitched = datalib.stitch_overlapping_patches(patches, x.shape, slack=slack)
    assert (x == stitched).all()


def test_sample_tensor_at_coordinates():
    Z = torch.zeros([2,5,100,90])
    Z[0,2,77,77] = 77.7
    Z[0,2, 0, 0] = 11.1
    Z[1,4,33,44] = 33.3
    kp = torch.as_tensor([
        [77,77],
        [44,33],
    ]).reshape(2,1,2)

    samples = datalib.sample_tensor_at_coordinates(Z, kp)
    assert samples.shape == (2,5,1)
    assert torch.all( samples[0,:,0] == Z[0,:,77,77] )
    assert torch.all( samples[1,:,0] == Z[1,:,33,44] )

    raised_error = 0
    try:
        # out of bounds
        kp2 = torch.as_tensor([
            [-1, -1],
            [10000, 33],
        ])
        should_fail = datalib.sample_tensor_at_coordinates(Z,kp2)
    except AssertionError:
        raised_error = 1

    assert raised_error

    # non-batched
    samples2 = datalib.sample_tensor_at_coordinates(Z[0], kp[0])
    assert torch.all(samples2 == samples[0])
    

def test_rot90_coordinates():
    H,W = [100,100]
    
    x    = torch.zeros([H,W])
    p    = torch.randint(0,100, (2,), )
    #p    = torch.tensor([ 80,10 ])
    
    x[p[1], p[0]] = 1
    
    for k in range(4):
        xrot = torch.rot90(x, k)
        prot = datalib.rot90_coordinates(p, (H,W), k)
        print(k, prot, torch.argwhere(xrot).numpy()[0][::-1])
        assert xrot[prot[1], prot[0]] == 1
        assert np.allclose(datalib.sample_tensor_at_coordinates(xrot[None,None], prot[None]), 1)
    


def test_adjust_coordinates_for_crop():
    # Test input
    c_xy = torch.tensor([[10, 20], [50, 40], [50, 60]], dtype=torch.float32)
    new_size = (100, 100)
    cropbox = (20, 20, 60, 40)  # left, top, width, height

    expected_output = torch.tensor([
        [-16.6667, 0], [50, 50], [50, 100]
    ])

    output = datalib.adjust_coordinates_for_crop(c_xy, cropbox, new_size)

    assert torch.allclose(output, expected_output)


def test_faster_unique_dim0():
    fn = torch.jit.script(datalib.faster_unique_dim0)
    x = torch.tensor([
        [0,1], [0,2], [0,3], [0,4], [0,1],
        [1,2], [1,3], [1,5], [1,2], [2,3], [2,4], [2,4]
    ])
    out = fn(x)
    expected = torch.tensor([
        [0,1], [0,2], [0,3], [0,4],
        [1,2], [1,3], [1,5],        [2,3], [2,4],
    ])
    assert (out == expected).all()

    fn2 = torch.jit.script(datalib.faster_unique_dim0_with_counts)
    out2, cnts = fn2(x)
    assert (out2 == out).all()
    expected_cnts = torch.tensor([2,1,1,1,2,1,1,1,2])
    assert (cnts == expected_cnts).all()

