import traininglib.segmentation.segmentationmodel as segm

import torch


def test_connected_components():
    a = torch.zeros([1,1,444,333])
    a[..., 5:15, 10:300] = 1
    a[..., 40:45, 20:70] = 1
    a[..., 50:90, 30:100] = 1

    b = segm.connected_components_max_pool(a.bool())
    assert b.shape == a.shape
    assert len(torch.unique(b)) == 4   # 3x blobs + zero





