import typing as tp
import argparse
import datetime
import glob
import os
import re
import time
import torch

def base_training_argparser() -> argparse.ArgumentParser:
    '''Construct an ArgumentParser with commonly used training parameters 
       for code re-use'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputsize', type=int,   default=512,  help='Default: 512px')
    parser.add_argument('--lr',        type=float, default=1e-3, help='Default: 1e-3')
    parser.add_argument('--epochs',    type=int,   default=30,   help='Default: 30')
    parser.add_argument('--batchsize', type=int,   default=8,    help='Default: 8')
    #TODO: warmup
    parser.add_argument(
        '--checkpointdir', 
        type    = time.strftime, 
        default = './checkpoints/%Y-%m-%d_%Hh%Mm%Ss', 
        help    = 'Where to save trained models'
    )
    parser.add_argument('--suffix', type=str, help='Suffix to append to checkpoint name')
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



def base_segmentation_training_argparser(
    pos_weight:float    = 1.0,
    margin_weight:float = 0.0,
) -> argparse.ArgumentParser:
    parser = base_training_argparser()
    parser.add_argument(
        '--trainsplit',
        required = True, 
        help     = 'Path to csv file containing input-target file pairs'
    )
    parser.add_argument(
        '--valsplit', 
        required = False, 
        help     = 'Path to csv file containing input-target file pairs'
    )
    parser.add_argument(
        '--pos-weight',    
        type    = float, 
        default = pos_weight,    
        help    = f'Default: {pos_weight}',
    )
    parser.add_argument(
        '--margin-weight', 
        type    = float, 
        default = margin_weight, 
        help    = f'Default: {margin_weight:.1f}',
    )
    parser.add_argument(
        '--rotate',
        action  = argparse.BooleanOptionalAction,
        default = True,
        help    = f'Rotate images during training'
    )
    return parser

MODEL_PLACEHOLDER  = '<model>'
LATEST_PLACEHOLDER = '<latest>'

class InferenceArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *a, **kw) -> argparse.Namespace: # type: ignore [override]
        args = super().parse_args(*a, **kw)
        
        model_name  = os.path.basename(args.model)
        args.output = args.output.replace(MODEL_PLACEHOLDER, model_name)
        return args



def validate_model_argument(x:str) -> str:
    '''If the argument is a folder containing a single .pt.zip, return this file.
       Replace placeholder <latest> with the latest trained model'''
    if os.path.basename(x) == LATEST_PLACEHOLDER:
        directory = os.path.dirname(x)
        contents  = os.listdir(directory)
        #reged for YYYY-MM-DD_HHh-MMm-SSs
        pattern   = r"^\d{4}-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s_.*"
        dated_folders = [
            item for item in contents if re.match(pattern, item)
        ]
        if len(dated_folders) == 0:
            raise FileNotFoundError(
                f'Could not find checkpoints in specified model directory {directory}'
            )

        sorted_folders = sorted(
            dated_folders, 
            key = lambda x: datetime.datetime.strptime(x[:20], '%Y-%m-%d_%Hh%Mm%Ss')
        )
        # continue searching for the .pt.zip below
        x = os.path.join(directory, sorted_folders[-1])
    
    if os.path.isdir(x):
        contents = glob.glob( os.path.join(x, '*.pt.zip') )
        if len(contents) == 0:
            raise FileNotFoundError(
                f'Specified model directory {x} does not contain a .pt.zip file'
            )
        if len(contents) > 1:
            raise FileNotFoundError(
                f'Specified model directory {x} contains multiple .pt.zip files'
            )
        x = contents[0]
    
    return x



def base_inference_argparser() -> argparse.ArgumentParser:
    '''Construct an ArgumentParser with commonly used inference parameters 
       for code re-use'''
    parser = InferenceArgumentParser()
    parser.add_argument(
        '--input', 
        required = True, 
        help     = 'Glob-like pattern to input images or path to csv split file',
    )
    parser.add_argument(
        '--model', 
        required = True,
        type     = validate_model_argument,
        help     = 'Path to .pt.zip model '\
            '(or to a folder containing a single .pt.zip).'\
            f'Placeholder {LATEST_PLACEHOLDER} uses last trained model.'
    )
    parser.add_argument(
        '--output', 
        default = f'./inference/{MODEL_PLACEHOLDER}', 
        help    = 'Where to save results. '\
            f'Placeholder {MODEL_PLACEHOLDER} replaced with model filename.',
    )
    parser.add_argument(
        '--device', 
        type    = cpu_or_gpu, 
        default = _default_device(), 
        help    = 'Which GPU to use (or CPU).',
    )
    return parser
