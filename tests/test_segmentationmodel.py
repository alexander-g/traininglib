import traininglib.segmentation.segmentationmodel as segm
import traininglib.segmentation.skeletonization   as skel

import torch


def test_connected_components():
    a = torch.zeros([1,1,444,333])
    a[..., 5:15, 10:300] = 1
    a[..., 40:45, 20:70] = 1
    a[..., 50:90, 30:100] = 1

    b = segm.connected_components_max_pool(a.bool())
    assert b.shape == a.shape
    assert len(torch.unique(b)) == 4   # 3x blobs + zero


def test_skeletonization():
    skeletonize = torch.jit.script(skel.skeletonize)
    
    x = torch.zeros([1,1,100,100])
    x[..., 5:10,   5:90] = 1
    x[..., 15:20, 20:70] = 1

    x_sk = skeletonize(x)

    assert torch.all(x_sk[0,0, :7,   :  ] == 0)
    assert torch.all(x_sk[0,0, 7,   7:87] == 1)
    assert torch.all(x_sk[0,0, 8:10, :  ] == 0)

