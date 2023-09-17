from traininglib import datalib
import tempfile, os
import numpy as np

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

