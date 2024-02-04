import typing as tp
import importlib
import copy
import io
import os
import tempfile
import zipfile

import torch
from . import onnxlib



TensorList = tp.List[torch.Tensor]
TensorDict = tp.Dict[str, torch.Tensor]
LossFunction = tp.Callable[..., torch.Tensor]

class ExportedTrainingStep(tp.NamedTuple):
    torchscriptmodule: torch.jit.ScriptModule
    trainingstate:     TensorDict

    def save_as_zipfile(self, path:str, inputfeed:TensorDict) -> None:
        if not path.endswith('.pt.zip'):
            path = f'{path}.pt.zip'
        base    = os.path.splitext(os.path.basename(path))[0]
        tmpdir  = tempfile.TemporaryDirectory()
        tmppath = os.path.join(tmpdir.name, 'training.torchscript')
        self.torchscriptmodule.save(tmppath)
        ts_bytes = open(tmppath, 'rb').read()
        with zipfile.ZipFile(path, 'w') as zipf:
            zipf.writestr(
                f'{base}/onnx/training.torchscript', ts_bytes
            )
            onnxlib.write_tensordict_to_zipfile(
                zipf, self.trainingstate, inputfeed, 'training.schema.json'
            )

def export_model_for_training(
    m:         torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):  
    m         = copy.deepcopy(m)
    optimizer = copy.deepcopy(optimizer)
    trainstepmodule = TrainStep(m, optimizer)
    #TODO: create a factory function rather than handling this in constructor
    #XXX:  isnt this the factory function?
    initial_trainingstate = trainstepmodule.initial_trainingstate
    #deleting parameters to reduce saved file size
    del trainstepmodule.initial_trainingstate
    #TODO:this one doesnt work yet
    #for p in trainstepmodule.parameters():
    #    p.data = torch.empty(0)
    trainstepscript = torch.jit.script(trainstepmodule)

    return ExportedTrainingStep(
        torchscriptmodule = trainstepscript,
        trainingstate     = initial_trainingstate,
    )



class TrainStep(torch.nn.Module):
    def __init__(
        self, 
        module:    torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        _check_module_annotations(module)
        self.module    = torch.jit.script(module)

        # trainable model parameters with gradients
        paramsdict = {k:p for k,p in self.named_parameters()}
        params:tp.List[torch.nn.Parameter] = list(paramsdict.values())

        self.get_gradients = torch.jit.script(_get_gradients)

        gradients:TensorList = self.get_gradients(params)
        self.paramkeys  = list( paramsdict.keys() )
        # modelstate: trainable and non-trainable parameters (requires_grad)
        self.modelstate = (
            {k:v for k,v in self.named_parameters()}
            | {k:v for k,v in self.named_buffers()}
        )
        self.optimizer  = onnxlib.initialize_optimizer(optimizer)

        gradsdict = dict(zip(self.paramkeys, gradients))
        optimizerstate:TensorDict = _pack_optimizerstate(self.optimizer.buffers)
        optimizerkeys: tp.List[str] = list(optimizerstate.keys())
        trainingstate = self.modelstate | optimizerstate
        trainingstate = {k:v.clone().detach() for k,v in trainingstate.items()}

        def optimizer_step(
            grads:         TensorDict, 
            trainingstate: TensorDict
        ) -> TensorDict:
            '''Helper function running an optimizer step. Can be traced.'''
            optimizerstate, modelstate = \
                 _extract_from_tensordict(trainingstate, optimizerkeys)
            optimizerbuffers = _unpack_optimizerstate(optimizerstate)
            modelparams, modelbuffers = \
                _extract_from_tensordict(modelstate, self.paramkeys)
            
            new_modelparams, new_optimizerbuffers = onnxlib.run_optimizer(
                grads, 
                modelparams, 
                optimizerbuffers, 
                self.optimizer.func, 
                self.optimizer.hyper_params
            )
            new_optimizerstate = _pack_optimizerstate(new_optimizerbuffers)
            new_trainingstate  = (
                new_modelparams | modelbuffers | new_optimizerstate
            )
            assert trainingstate.keys() == new_trainingstate.keys()
            return new_trainingstate
        self.optimizer_fxt = torch.jit.trace(
            optimizer_step, (gradsdict, trainingstate), strict=False
        )
 
        self.initial_trainingstate = trainingstate
        self.trainingstate_keys = list(trainingstate.keys())

    
    def _prepare_forward(self, trainingstate:TensorDict) -> TensorDict:
        # zero_grad() does not work with torchscript
        params = _extract_from_tensordict(self.modelstate, self.paramkeys)[0]
        for g in self.get_gradients(params.values()):
            g *= 0
        
        # set parameters from input
        old_modelstate = {k:p.clone() for k,p in self.modelstate.items()}
        self._set_modelstate(trainingstate)
        return old_modelstate
    
    def _set_modelstate(self, new_state:TensorDict) -> None:
        with torch.no_grad():
            for k,p in self.modelstate.items():
                #p.ravel()[:] = new_state[k].ravel()
                torch.ops.aten.set_data(p, new_state[k])

    def _backward_step(
        self, 
        loss:          torch.Tensor, 
        trainingstate: TensorDict
    ) -> TensorDict:
        loss.backward()
        params = _extract_from_tensordict(self.modelstate, self.paramkeys)[0]
        grads = [
             # ensure new tensor
             torch.as_tensor(g).clone() for g in self.get_gradients(params.values())
        ]
        gradsdict = dict(zip(self.paramkeys, grads))

        new_trainingstate = self.optimizer_fxt(gradsdict, trainingstate)

        output = dict(new_trainingstate)
        output.update( {'loss': loss} )
        output.update( {f'{k}.gradient':v for k,v in gradsdict.items()} )
        return output
    
    def forward(self, inputfeed:TensorDict) -> TensorDict:
        trainingstate, inputs = \
            _extract_from_tensordict(inputfeed, self.trainingstate_keys)
        old_modelstate = self._prepare_forward(trainingstate)
        loss, outputs  = self.module(inputs)
        output         = self._backward_step(loss, trainingstate)
        self._set_modelstate(old_modelstate)
        return output


def _pack_optimizerstate(buffers:tp.Dict[str, TensorList]) -> TensorDict:
    '''Pack optimizer buffers into a flat dict'''
    optimizerstate:TensorDict = {}

    # buffers are inserted from groupname and index
    for buffergroupname, buffergroup in buffers.items():
        for i, buffer in enumerate(buffergroup):
            optimizerstate[f'{buffergroupname}.{i}.buffer'] = buffer
    return optimizerstate

def _unpack_optimizerstate(optimizerstate:TensorDict) -> tp.Dict[str,TensorList]:
    '''Unpack a flat dict returned from `_pack_optimizerstate()`'''
    buffers: tp.Dict[str, TensorList] = {}
    for k,v in optimizerstate.items():
        if k.endswith('.buffer'):
            buffergroupname = '.'.join(k.split('.')[:-2])
            buffers[buffergroupname] = buffers.get(buffergroupname, []) + [v]
    return buffers

def _extract_from_tensordict(
    tensordict: TensorDict,
    keys:       tp.List[str],
) -> tp.Tuple[TensorDict, TensorDict]:
    '''Split a dict, returning one with specified keys and remaining.'''
    in_dict:TensorDict  = {}
    out_dict:TensorDict = {}
    for k,v in tensordict.items():
        if k in keys:
            in_dict[k] = v
        else:
            out_dict[k] = v
    return in_dict, out_dict


def _get_gradients(params:TensorList) -> TensorList:
    '''Return gradients from parameters, creating new tensors if needed.
       torch.jit.script()-able.'''
    gradients:TensorList = []
    for p in params:
        g = p.grad
        if g is None or g.dtype==25:
            #don't know what exactly dtype 25 is but it seems to be the
            #c++/torchscript equivalent of None
            g = torch.zeros_like(p)
        gradients.append(g)
    return gradients

def _check_module_annotations(m:torch.nn.Module) -> None:
    annotations = m.forward.__annotations__
    return_type = tp.Tuple[torch.Tensor, TensorDict]
    if 'return' not in annotations or annotations['return'] != return_type:
        raise Exception(
            f'Module must have an annotated return type {return_type}'
        )
    
    input_type = TensorDict
    if len(annotations) > 2:
        raise Exception(f'Module must have a single input {input_type}')
    
    input_name = [k for k in annotations.keys() if k != 'return'][0]
    if annotations[input_name] != input_type:
        raise Exception(f'Only {input_type} supported as module input')


def write_tensordict_to_zipfile(path:str, x:TensorDict):
    with zipfile.ZipFile(path, 'w') as zipf:
        onnxlib.write_tensordict_to_zipfile(zipf, x)

def pack_tensordict(x:TensorDict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zipf:
        onnxlib.write_tensordict_to_zipfile(zipf, x)
    return buffer.getvalue()
