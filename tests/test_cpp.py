import typing as tp
import ctypes
import os
import tempfile

import torch
from traininglib import torchscriptlib

LIB_PATH = './cpp/build/libTSinterface.so'


class Module(torch.nn.Module):
    def __init__(self, basemodule):
        super().__init__()
        self.basemodule = basemodule
    
    def forward(self, inputfeed:tp.Dict[str, torch.Tensor]):
        return self.basemodule(inputfeed['x'])

def test_initialize_module():
    assert os.path.exists(LIB_PATH), 'C++ interface has not been built'

    lib = ctypes.CDLL(LIB_PATH)

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


    rc = lib.run_module(invalid_bytes, len(invalid_bytes))
    assert rc != 0, 'Running module with invalid data should not succeed'

    x = torch.ones([1,3,4,4])*2
    inputs = torchscriptlib.pack_tensordict({'x': x})
    rc     = lib.run_module(inputs, len(inputs))
    assert rc == 0, 'Running module failed'

    y = m({'x':x})

    assert 0
