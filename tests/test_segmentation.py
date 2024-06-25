import os
import tempfile

import PIL.Image
import torch

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

    ds = segmentation.PatchedCachingDataset([
        (os.path.join(tempdir.name, 'im0.jpg'), os.path.join(tempdir.name, 'an0.png')),
        (os.path.join(tempdir.name, 'im1.png'), os.path.join(tempdir.name, 'an1.png')),
    ], patchsize=100, cachedir=os.path.join(tempdir.name, 'cache'))

    ld = datalib.create_dataloader(ds, batch_size=2)
    #assert no exception
    assert len(list(ld)) == 1



def test_margin_loss_multilabel():
    y5 = torch.zeros([2,3,50,50])
    t5 = torch.zeros([2,3,50,50]).bool()

    loss = segmentation.margin_loss_multilabel(y5,t5, logits=True)
    assert torch.isfinite(loss)


    y5[1,0,1:] = 1
    y5[0,2,1:] = 1
    t5[1,0,1:] = 1
    t5[0,2,1:] = 1
    loss = segmentation.margin_loss_multilabel(y5,t5, logits=True)
    # not zero because logits = True
    assert loss > 0

    loss = segmentation.margin_loss_multilabel(y5,t5, logits=False)
    # should be zero, esp should not take samples inbetween batches / channels
    assert loss == 0

    loss = segmentation.margin_loss_fn(y5[:,:1],t5[:,:1])
    # TODO: fix margin_loss_fn()
    #assert loss == 0
