import traininglib.segmentation.segmentationmodel as segm
import traininglib.segmentation.connectedcomponents as concom
from util import _export_to_onnx

import io
import os
import tempfile

import onnxruntime as ort
import torch
import numpy as np



def test_connected_components():
    a = torch.zeros([1,1,444,333])
    a[..., 5:15, 10:300] = 1
    a[..., 40:45, 20:70] = 1
    a[..., 50:90, 30:100] = 1

    b = concom.connected_components_max_pool(a.bool())
    assert b.shape == a.shape
    assert len(torch.unique(b)) == 4   # 3x blobs + zero


def test_connected_components_patchwise():
    a = torch.zeros([1,1,200,300])
    #component 1
    a[...,   5: 10, :]    = 1
    a[..., 105:110, :]    = 1
    a[...,   5:110, 290:] = 1
    #component 2
    a[..., 150:160, 50:160] = 2

    b = concom.connected_components_patchwise(a.bool(), patchsize=100)
    assert b.shape == a.shape
    assert len(torch.unique(b)) == len(torch.unique(a)) # 2x components + zero
    
    @torch.jit.script
    def concom_pw100(x:torch.Tensor) -> torch.Tensor:
        return concom.connected_components_patchwise(x, patchsize=100)
    session = _export_to_onnx(
        concom_pw100, 
        args=(torch.zeros([1,1,64,64]).bool(), )
    )

    b_onnx = session.run(None, {'x':a.bool().numpy()})
    assert np.all(b_onnx == b.numpy())


def test_adjacency_dfs():
    adj = torch.as_tensor([
        [1,2], #1
        [3,4], #2
        [5,6], #3
        [7,1], #1
        [8,2], #1
    ])

    labeled = concom.connected_components_from_adjacency_list(adj)
    assert len(torch.unique(labeled)) == 3


def test_segmentationmodel_export():
    conv = torch.nn.Conv2d(3,1, kernel_size=3)
    m = segm.SegmentationModel_ONNX(
        inputsize=512, 
        classes=[], 
        module=conv, 
        patchify=True,
        connected_components=True,
        skeletonize=True,
        #paths=True,
    )
    x = torch.ones([1,1024,1024,3]).byte()
    i = torch.tensor(0)
    y = torch.ones([1,1,1,1])
    outputs = m(x,i,y)
    assert len(outputs) == len(m.output_names)

    tempdir = tempfile.TemporaryDirectory()
    tempf   = os.path.join(tempdir.name, 'exported')
    m.export_to_onnx(tempf)


