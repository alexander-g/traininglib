import typing as tp
import ctypes
import os
import sys
import tempfile
import io
import json
import zipfile

import numpy as np
import torch, torchvision
import pytest

from traininglib import torchscriptlib
from traininglib import ts_cpp_interface as ts_cpp

LIB_PATH = (
    './cpp/build/libTSinterface.so' if 'win32' not in sys.platform else 
    './cpp/build/Release/TSinterface.dll'
)



TensorDict = torchscriptlib.TensorDict




class TestItem(tp.NamedTuple):
    module:    torch.nn.Module
    inputfeed: TensorDict
    desc:      str


testdata = [
    TestItem(
        module    = ts_cpp.BasicModule(torch.nn.Conv2d(3,1,kernel_size=3)),
        inputfeed = {'x':torch.rand(2,3,64,64)},
        desc      = 'basic-conv'
    ),

    TestItem(
        module = ts_cpp.DetectionModule(
            torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=None, progress=False, weights_backbone=None
            ).eval()
        ),
        inputfeed = {'x':torch.rand(1,3,32,32)},
        desc      = 'faster-rcnn.eval'
    ),

    TestItem(
        module = ts_cpp.DetectionModule(
            torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=None, progress=False, weights_backbone=None
            ).train()
        ),
        inputfeed = {
            'x':        torch.rand(1,3,32,32),
            't.boxes':  torch.as_tensor([[ 0.0, 0.0, 5.0, 5.0 ]]),
            't.labels': torch.as_tensor([1]),
        },
        desc = 'faster-rcnn.train'
    ),
]


@pytest.mark.parametrize("testitem", testdata)
def test_invalid_inputs(testitem):
    print(f'==========TRAINING TEST: {testitem.desc}==========')
    lib = ts_cpp.load_library(LIB_PATH)

    invalid_bytes = b'banana'
    rc = lib.initialize_module(invalid_bytes, len(invalid_bytes))
    assert rc != 0, 'Initialization with invalid data should not succeed'


    m  = ts_cpp.BasicModule(torch.nn.Conv2d(3,1,kernel_size=3))
    ms = torch.jit.script(m)
    
    m_bytes = ts_cpp.export_module_to_torchscript_bytes(m)
    rc = lib.initialize_module(m_bytes, len(m_bytes))
    assert rc == 0, 'Initialization failed'


    output_pointers = ts_cpp.create_output_pointers()
    rc = lib.run_module(
        invalid_bytes, len(invalid_bytes), *output_pointers, False
    )
    assert rc != 0, 'Running module with invalid data should not succeed'



@pytest.mark.parametrize("testitem", testdata)
def test_run_module(testitem):
    print(f'==========TRAINING TEST: {testitem.desc}==========')
    m   = testitem.module
    lib = ts_cpp.TS_CPP_Module.initialize_from_module(LIB_PATH, m)
    assert not isinstance(lib, Exception)

    ms  = torch.jit.script(m)

    inputfeed = testitem.inputfeed
    torch.manual_seed(0)
    eager_output = ms(inputfeed)

    torch.manual_seed(0)
    inputs = torchscriptlib.pack_tensordict(inputfeed)
    output = lib.run(inputfeed)
    assert not isinstance(output, Exception), output

    for key in eager_output.keys():
        assert torch.allclose(eager_output[key], output[key])


    #assert 0


def test_run_module_from_ptzip():
    from traininglib.segmentation import segmentationmodel
    segm = segmentationmodel.SegmentationModel_ONNX(
        module    = torch.nn.Conv2d(3,1,kernel_size=3, padding=1),
        inputsize = 64,
        classes   = [],
        patchify   = True,
    )
    m = segmentationmodel.SegmentationModel_TorchScript(segm)
    tempdir = tempfile.TemporaryDirectory()
    tempf   = os.path.join(tempdir.name, 'module.pt.zip')
    m.export_to_torchscript(tempf)

    lib = ts_cpp.TS_CPP_Module.initialize_from_ptzip(LIB_PATH, tempf)
    assert not isinstance(lib, Exception)

    inputfeed = {
        'x': torch.ones([1,128,128,3], dtype=torch.uint8),
        'i': torch.tensor(0),
        'y': torch.ones([1,1,1,1]),
    }
    output = lib.run(inputfeed)
    assert not isinstance(output, Exception), output


