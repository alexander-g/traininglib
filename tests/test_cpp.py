import typing as tp
import ctypes
import os
import tempfile
import io
import json
import zipfile

import numpy as np
import torch
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
                    schema = json.loads(zipf.read(p))
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

class Module(torch.nn.Module):
    def __init__(self, basemodule):
        super().__init__()
        self.basemodule = basemodule
    
    def forward(self, inputfeed:tp.Dict[str, torch.Tensor]):
        return {'y': self.basemodule(inputfeed['x'])}

def test_initialize_module():
    lib = load_library(LIB_PATH)

    invalid_bytes = b'banana'
    rc = lib.initialize_module(invalid_bytes, len(invalid_bytes))
    assert rc != 0, 'Initialization with invalid data should not succeed'


    m  = Module(torch.nn.Conv2d(3,1, kernel_size=3))
    ms = torch.jit.script(m)
    tempdir  = tempfile.TemporaryDirectory()
    temppath = os.path.join(tempdir.name, 'module.torchscript')
    ms.save(temppath)
    m_bytes  = open(temppath, 'rb').read()

    
    rc = lib.initialize_module(m_bytes, len(m_bytes))
    assert rc == 0, 'Initialization failed'


    output_pointers = create_output_pointers()
    rc = lib.run_module(invalid_bytes, len(invalid_bytes), *output_pointers)
    assert rc != 0, 'Running module with invalid data should not succeed'

    x = torch.ones([1,3,4,4])*2
    inputs = torchscriptlib.pack_tensordict({'x': x})
    rc     = lib.run_module(inputs, len(inputs), *output_pointers)
    assert rc == 0, 'Running module failed'
    output_bytes = output_pointers_to_bytes(*output_pointers)
    output       = output_bytes_to_tensordict(output_bytes)
    assert not isinstance(output, Exception), output

    lib.free_memory(output_pointers[0]._obj)

    eager_output = m({'x':x})
    assert torch.allclose(eager_output['y'], output['y'])

    #assert 0
