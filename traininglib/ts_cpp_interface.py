import typing as tp
import ctypes
import io
import json
import os
import tempfile
import zipfile

import numpy as np
import torch

from . import torchscriptlib as tslib

TensorDict = tslib.TensorDict


class BasicModule(torch.nn.Module):
    def __init__(self, basemodule):
        super().__init__()
        self.basemodule = basemodule
    
    def forward(self, inputfeed:TensorDict):
        return {'y': self.basemodule(inputfeed['x'])}


class DetectionModule(BasicModule):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.basemodule._has_warned = True  #pytorch is annoying

    def _forward(self, inputfeed:TensorDict):
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
    
    def forward(self, inputfeed:TensorDict):
        return self._forward(inputfeed)

class DetectionTrainStepModule(DetectionModule):
    def forward(self, x:TensorDict) -> tp.Tuple[torch.Tensor, TensorDict]:
        losses = self._forward(x)
        loss: torch.Tensor = (
            losses['loss_box_reg']
            + losses['loss_classifier']
            + losses['loss_objectness']
            + losses['loss_rpn_box_reg']
        )
        logs:TensorDict = losses
        return loss, logs



class TS_CPP_Module:
    @classmethod
    def initialize_from_module(
        cls, 
        path_to_lib: str, 
        module:      torch.nn.Module,
    ) -> tp.Union["TS_CPP_Module", Exception]:
        ts_bytes = export_module_to_torchscript_bytes(module)
        return cls.initialize_from_torchscript_bytes(path_to_lib, ts_bytes)
    
    @classmethod
    def initialize_from_ptzip(
        cls, 
        path_to_lib:   str, 
        path_to_ptzip: str,
    ) -> tp.Union["TS_CPP_Module", Exception]:
        with zipfile.ZipFile(path_to_ptzip) as zipf:
            ts_files = [f for f in zipf.namelist() if f.endswith('.torchscript')]
            if len(ts_files) == 0:
                return Exception('Zipfile does not contain torschript files')
            if len(ts_files) > 1:
                return Exception('Zipfile contains multiple torschript files')
            ts_bytes = zipf.read(ts_files[0])
        return cls.initialize_from_torchscript_bytes(path_to_lib, ts_bytes)

    @classmethod
    def initialize_from_torchscript_bytes(
        cls, 
        path_to_lib:   str, 
        ts_bytes:      bytes,
    ) -> tp.Union["TS_CPP_Module", Exception]:
        try:
            lib = load_library(path_to_lib)
        except Exception as e:
            return e
        
        rc = lib.initialize_module(ts_bytes, len(ts_bytes))
        if rc != 0:
            return Exception('Failed to initialize torchscript module')
        return cls(lib)
    
    def __init__(self, lib:ctypes.CDLL):
        self.lib = lib
    
    def run(self, inputfeed:tslib.TensorDict) -> tslib.TensorDict|Exception:
        inputs:bytes = tslib.pack_tensordict(inputfeed)
        output_pointers = create_output_pointers()
        rc = self.lib.run_module(inputs, len(inputs), *output_pointers, True)
        if rc != 0:
            return Exception('Failed to run module')
        output_bytes = output_pointers_to_bytes(*output_pointers)
        output       = output_bytes_to_tensordict(output_bytes)
        self.lib.free_memory(output_pointers[0]._obj)
        return output




def load_library(path:str):
    '''Load `libTSinterface.so` compiled from `cpp/interface.cpp`
       and assign types to functions.  '''
    assert os.path.exists(path), 'C++ interface library does no exist'

    lib = ctypes.CDLL(path)
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


def export_module_to_torchscript_bytes(module:torch.nn.Module) -> bytes:
    '''Run torch.jit.script() on a module, save to file, return as bytes.'''
    scripted = torch.jit.script(module)
    tempdir  = tempfile.TemporaryDirectory()
    temppath = os.path.join(tempdir.name, 'module.torchscript')
    scripted.save(temppath)
    ts_bytes = open(temppath, 'rb').read()
    return ts_bytes



def create_output_pointers() -> tp.Tuple:
    data_p = ctypes.byref(ctypes.POINTER(ctypes.c_uint8)())
    size_p = ctypes.byref(ctypes.c_size_t())
    return data_p, size_p

def output_pointers_to_bytes(data_p, size_p) -> bytes:
    return ctypes.string_at(data_p._obj, size_p._obj.value)

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

