import ctypes
import os
import tempfile

import torch

LIB_PATH = './cpp/build/libtestlib.so'


def test_initialize_module():
    assert os.path.exists(LIB_PATH), 'C++ interface has not been built'

    lib = ctypes.CDLL(LIB_PATH)

    invalid_bytes = b'banana'
    rc = lib.initialize_module(invalid_bytes, len(invalid_bytes))
    assert rc != 0, 'Initialization with invalid data should not succeed'


    m  = torch.nn.Conv2d(3,1, kernel_size=3)
    ms = torch.jit.script(m)
    tempdir  = tempfile.TemporaryDirectory()
    temppath = os.path.join(tempdir.name, 'module.torchscript')
    ms.save(temppath)
    m_bytes  = open(temppath, 'rb').read()
    
    rc = lib.initialize_module(m_bytes, len(m_bytes))
    assert rc == 0, 'Initialization failed'
