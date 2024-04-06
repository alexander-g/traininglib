import os
import tempfile

import PIL.Image

from traininglib import segmentation, datalib


def test_small_imagesizes():
    '''Actual bug: if images are smaller than twice the patchsize,
       and unequal among each other, this will leaed to an exception 
       in the default collate function.'''


    tempdir = tempfile.TemporaryDirectory()
    PIL.Image.new('RGB', (50,50)).save(os.path.join(tempdir.name, 'im0.jpg'))
    PIL.Image.new('L',   (35,65)).save(os.path.join(tempdir.name, 'im1.png'))
    PIL.Image.new('L',   (50,50)).save(os.path.join(tempdir.name, 'an0.png'))
    PIL.Image.new('L',   (35,65)).save(os.path.join(tempdir.name, 'an1.png'))

    ds = segmentation.SegmentationDataset([
        (os.path.join(tempdir.name, 'im0.jpg'), os.path.join(tempdir.name, 'an0.png')),
        (os.path.join(tempdir.name, 'im1.png'), os.path.join(tempdir.name, 'an1.png')),
    ], patchsize=100, cachedir=os.path.join(tempdir.name, 'cache'))

    ld = datalib.create_dataloader(ds, batch_size=2)
    #assert no exception
    assert len(list(ld)) == 1

