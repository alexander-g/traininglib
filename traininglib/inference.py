import os
import typing as tp

from . import args, datalib, modellib


class InferenceItem(tp.NamedTuple):
    output:    tp.Any
    inputfile: str


def base_inference(args:args.Namespace) -> tp.Generator[InferenceItem,None,None]:
    '''Generator that loads a model and inputs and yields outputs'''
    inputs = datalib.collect_inputfiles(args.input)
    model  = modellib.load_model(args.model).to(args.device)
    os.makedirs(args.output, exist_ok=True)
    
    print(f'Running inference on {len(inputs)} files.')
    print(f'Using model {args.model}')
    print(f'Saving output to {args.output}')
    for i, imagefile in enumerate(inputs):
        print(f'[{i:4d}/{len(inputs)}]', end='\r')
        output = model.process_image(imagefile)
        yield InferenceItem(output, imagefile)
    print()

