import typing as tp
import ctypes
import os
import tempfile
import io
import json
import zipfile

import numpy as np
import torch, torchvision
import pytest

from traininglib import torchscriptlib

LIB_PATH = './cpp/build/libTSinterface.so'



def load_library(path:str):
    assert os.path.exists(LIB_PATH), 'C++ interface library does no exist'

    lib = ctypes.CDLL(LIB_PATH)
    lib.initialize_module.restype  = ctypes.c_int32
    lib.initialize_module.argtypes = [
        ctypes.c_char_p,                                #inputbuffer
        ctypes.c_size_t,                                #inputbuffersize
    ]

    lib.run_module.restype  = ctypes.c_int32
    lib.run_module.argtypes = [
        ctypes.c_char_p,                                #inputbuffer
        ctypes.c_size_t,                                #inputbuffersize
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), #outputbuffer
        ctypes.POINTER(ctypes.c_size_t),                #outputbuffersize
        ctypes.c_bool,                                  #debug
    ]

    lib.free_memory.restype  = None
    lib.free_memory.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    return lib


def create_output_pointers() -> tp.Tuple:
    data_p = ctypes.byref(ctypes.POINTER(ctypes.c_uint8)())
    size_p = ctypes.byref(ctypes.c_size_t())
    return data_p, size_p

def output_pointers_to_bytes(data_p, size_p) -> bytes:
    return bytes(data_p._obj[:size_p._obj.value])


TensorDict = tp.Dict[str, torch.Tensor]

def output_bytes_to_tensordict(data:bytes) -> TensorDict|Exception:
    buffer = io.BytesIO(data)
    with zipfile.ZipFile(buffer) as zipf:
        paths = zipf.namelist()
        for p in paths:
            if p.endswith('/inference.schema.json'):
                try:
                    jsonraw = zipf.read(p)
                    schema  = json.loads(jsonraw)
                    assert isinstance(schema, dict)
                    break
                except Exception as e:
                    return Exception('Could not read inference schema')
        else:
            return Exception('Could not find inference schema')
        
        result:TensorDict = {}
        for key,schema_item in schema.items():
            if (
                   'path'  not in schema_item 
                or 'shape' not in schema_item 
                or 'dtype' not in schema_item
            ):
                return Exception('Invalid schema')
            
            try:
                data   = zipf.read(schema_item['path'])
                # NOTE: not using torch.frombuffer because fuck torch
                array  = np.frombuffer(data, schema_item['dtype']).copy()
                tensor = torch.as_tensor(array).reshape(schema_item['shape'])
            except Exception as e:
                return e
            
            result[key] = tensor
    return result



class BasicModule(torch.nn.Module):
    def __init__(self, basemodule):
        super().__init__()
        self.basemodule = basemodule
    
    def forward(self, inputfeed:tp.Dict[str, torch.Tensor]):
        return {'y': self.basemodule(inputfeed['x'])}


class DetectionModule(BasicModule):
    def forward(self, inputfeed:tp.Dict[str, torch.Tensor]):
        x = [inputfeed['x'][0]]
        if 't.boxes' in inputfeed and 't.labels' in inputfeed:
            t = [{
                'boxes':  inputfeed['t.boxes'],
                'labels': inputfeed['t.labels'],
            }]
        else:
            t = None
        losses, outputs = self.basemodule(x, t)
        return losses if self.basemodule.training else outputs[0]



class TestItem(tp.NamedTuple):
    module:    torch.nn.Module
    inputfeed: TensorDict
    desc:      str


testdata = [
    TestItem(
        module    = BasicModule(torch.nn.Conv2d(3,1,kernel_size=3)),
        inputfeed = {'x':torch.rand(2,3,64,64)},
        desc      = 'basic-conv'
    ),

    TestItem(
        module = DetectionModule(
            torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=None, progress=False, weights_backbone=None
            ).eval()
        ),
        inputfeed = {'x':torch.rand(1,3,32,32)},
        desc      = 'faster-rcnn.eval'
    ),

    TestItem(
        module = DetectionModule(
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
def test_initialize_module(testitem):
    print(f'==========TRAINING TEST: {testitem.desc}==========')
    lib = load_library(LIB_PATH)

    invalid_bytes = b'banana'
    rc = lib.initialize_module(invalid_bytes, len(invalid_bytes))
    assert rc != 0, 'Initialization with invalid data should not succeed'


    m  = testitem.module
    ms = torch.jit.script(m)
    tempdir  = tempfile.TemporaryDirectory()
    temppath = os.path.join(tempdir.name, 'module.torchscript')
    ms.save(temppath)
    m_bytes  = open(temppath, 'rb').read()

    
    rc = lib.initialize_module(m_bytes, len(m_bytes))
    assert rc == 0, 'Initialization failed'


    output_pointers = create_output_pointers()
    rc = lib.run_module(
        invalid_bytes, len(invalid_bytes), *output_pointers, False
    )
    assert rc != 0, 'Running module with invalid data should not succeed'

    inputfeed = testitem.inputfeed
    torch.manual_seed(0)
    eager_output = ms(inputfeed)

    torch.manual_seed(0)
    inputs = torchscriptlib.pack_tensordict(inputfeed)
    rc     = lib.run_module(inputs, len(inputs), *output_pointers, True)
    assert rc == 0, 'Running module failed'
    output_bytes = output_pointers_to_bytes(*output_pointers)
    output       = output_bytes_to_tensordict(output_bytes)
    assert not isinstance(output, Exception), output

    lib.free_memory(output_pointers[0]._obj)

    for key in eager_output.keys():
        assert torch.allclose(eager_output[key], output[key])


    #assert 0
