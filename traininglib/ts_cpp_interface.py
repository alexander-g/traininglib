import typing as tp
import ctypes
import os

import torch


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




