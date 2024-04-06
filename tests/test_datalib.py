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
    assert output[1][1] == f'{tmpdir.name}/{subfolder}/mask1.png'


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
