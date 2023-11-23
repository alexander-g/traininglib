import typing as tp
import time
import os
import shutil
import sys

import torch

from . import modellib



def generate_output_name(args) -> tp.Tuple[str,str]:
    '''Construct a standardized name, containing current time and some config parameters'''
    date = time.strftime('%Y-%m-%d')
    tod  = time.strftime('%Hh%Mm%Ss')
    sufx = (f'_{args.suffix}' if args.suffix else '')
    path = f'{args.checkpointdir}_{args.inputsize}px_{args.epochs}e{sufx}'
    name = f'{date}_{sufx}'
    return path, name

def collect_loaded_non_venv_modules() -> tp.List[str]:
    modules:tp.List[str] = []
    for name, module in sys.modules.items():
        modulefile = getattr(module, '__file__', None)
        if not isinstance(modulefile, str):
            continue
        
        if not modulefile.startswith(sys.prefix) and not 'site-packages' in modulefile:
            modules.append(name)
    return modules

def backup_code(destination:str) -> str:
    '''Save source at the destination folder, for reproducibility.'''
    destination = time.strftime(destination)
    cwd      = os.path.realpath(os.getcwd())+'/'
    srcfiles = collect_loaded_non_venv_modules()
    for src_f in srcfiles:
        src_f = os.path.realpath(src_f)
        dst_f = os.path.join(destination, src_f.replace(cwd, ''))
        os.makedirs(os.path.dirname(dst_f), exist_ok=True)
        shutil.copy(src_f, dst_f)
    open(os.path.join(destination, 'args.txt'), 'w').write(' '.join(sys.argv))
    return destination

def prepare_for_training(model, args, config) -> tp.Tuple[torch.nn.Module, str, str]:
    '''Some housekeeping tasks before training starts'''
    destination, name = generate_output_name(args, config)
    print('Output directory:', destination)
    backup_code(destination)

    #save already now and immediately reload
    #to avoid inconsistencies if the source code changes during training
    modelpath       = os.path.join(destination, name)
    modelpath_tmp   = model.save(f'{modelpath}.tmp')
    model           = modellib.load_model( modelpath_tmp )
    return model, modelpath, modelpath_tmp
