import typing as tp
import argparse
import time
import torch

def base_training_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputsize', type=int,   default=512,  help='Default: 512px')
    parser.add_argument('--lr',        type=float, default=1e-3, help='Default: 1e-3')
    parser.add_argument('--epochs',    type=int,   default=30,   help='Default: 30')
    parser.add_argument('--batchsize', type=int,   default=8,    help='Default: 8')
    #TODO: warmup
    parser.add_argument(
        '--checkpointdir', 
        type    = time.strftime, 
        default = './checkpoints/%Y-%m-%d_%Hh%Mm%Ss/', 
        help    = 'Where to save trained models'
    )
    parser.add_argument(
        '--device', type=cpu_or_gpu, default=_default_device(), help='Which GPU to use (or CPU).'
    )
    return parser


def cpu_or_gpu(x:str) -> str:
    if x == 'cpu':
        return 'cpu'
    else:
        if x.startswith('cuda:'):
            x = x[len('cuda:'):]
        gpu = int(x)
        if gpu < 0:
            raise ValueError('Non-negative GPU number required')
        if gpu > torch.cuda.device_count():
            raise ValueError('Device number too large')
        return f'cuda:{gpu}'

def _default_device() -> str:
    return ( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
