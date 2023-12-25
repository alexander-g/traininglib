import typing as tp
import importlib
import os
import tempfile
import uuid

import torch
from . import onnxlib



TensorList = tp.List[torch.Tensor]
TensorDict = tp.Dict[str, torch.Tensor]
LossFunction = tp.Callable[..., torch.Tensor]

class ExportedTrainingStep(tp.NamedTuple):
    torchscriptmodule: torch.jit.ScriptModule
    optimizerstate:    TensorDict

def export_model_for_training(
    m:         torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    subclasstuple = error = subclass_via_tempfile(m)
    if isinstance(subclasstuple, Exception):
        return error
    TrainStepClass, _tempdir = subclasstuple
    
    trainstepmodule = TrainStepClass(m, optimizer)
    trainstepscript = torch.jit.script(trainstepmodule)

    return ExportedTrainingStep(
        torchscriptmodule = trainstepscript,
        optimizerstate    = trainstepmodule.initial_optimizerstate,
    )



class TrainStep(torch.nn.Module):
    def __init__(
        self, 
        module:    torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.module    = torch.jit.script(module)
        self.params    = list(self.module.parameters())
        for p in self.params:
            p.grad = torch.zeros_like(p)
        self.grads:TensorList = [
            tp.cast(torch.Tensor, p.grad) for p in self.params
        ]
        self.paramkeys = list( dict(self.module.named_parameters()).keys() )
        self.optimizer = onnxlib.initialize_optimizer(optimizer)

        gradsdict  = dict(zip(self.paramkeys, self.grads))
        paramsdict = {
            k:v.clone().detach() for k,v in zip(self.paramkeys, self.params)
        }
        optimizerstate:tp.Dict[str, torch.Tensor] = \
            _pack_optimizerstate(paramsdict, self.optimizer.buffers)

        def optimizer_step(
            grads:          TensorDict, 
            optimizerstate: TensorDict
        ) -> TensorDict:
            '''Helper function running an optimizer step. Can be traced.'''
            params, buffers = _unpack_optimizerstate(optimizerstate)
            
            new_params, new_buffers = onnxlib.run_optimizer(
                grads, 
                params, 
                buffers, 
                self.optimizer.func, 
                self.optimizer.hyper_params
            )
            new_optimizerstate = _pack_optimizerstate(new_params, new_buffers)
            assert optimizerstate.keys() == new_optimizerstate.keys()
            return new_optimizerstate
        self.optimizer_fxt = torch.jit.trace(
            optimizer_step, (gradsdict, optimizerstate), strict=False
        )

        self._params_from_optimizerstate = torch.jit.trace(
            _unpack_params_from_optimizerstate, (optimizerstate,), strict=False
        )

        self.initial_optimizerstate = optimizerstate
        #self.forward = self._forward
    
    def _prepare_forward(self, optimizerstate:TensorDict) -> TensorList:
        # zero_grad() does not work with torchscript
        for g in self.grads:
            g[:] = torch.zeros_like(g)
        # set parameters from input
        old_params = [p.clone() for p in self.params]
        new_params = self._params_from_optimizerstate(optimizerstate)
        self._set_parameters(new_params)
        return old_params
    
    def _set_parameters(self, new_params:TensorList) -> None:
        with torch.no_grad():
            for i,p in enumerate(self.params):
                p[:] = new_params[i]

    def _backward_step(self, loss:torch.Tensor, optimizerstate:TensorDict) -> TensorDict:
        loss.backward()
        grads = [
            # ensure new tensor
            torch.as_tensor(p.grad).clone() for p in self.params
        ]
        gradsdict = dict(zip(self.paramkeys, grads))

        new_optimizerstate  = self.optimizer_fxt(gradsdict, optimizerstate)

        output = dict(new_optimizerstate)
        output.update( {'loss': loss} )
        output.update( {f'{k}.gradient':v for k,v in gradsdict.items()} )
        return output


def _pack_optimizerstate(
    params:TensorDict, 
    buffers:tp.Dict[str, TensorList]
) -> TensorDict:
    '''Pack parameters and optimizer buffers into a flat dict'''
    # params are used as is
    optimizerstate:TensorDict = dict(params)

    # buffers are inserted from groupname and index
    for buffergroupname, buffergroup in buffers.items():
        for i, buffer in enumerate(buffergroup):
            optimizerstate[f'{buffergroupname}.{i}.buffer'] = buffer
    return optimizerstate

def _unpack_optimizerstate(optimizerstate:TensorDict) \
-> tp.Tuple[TensorDict, tp.Dict[str, TensorList]]:
    '''Unpack a flat dict returned from `_pack_optimizerstate()`'''
    params:  TensorDict               = {}
    buffers: tp.Dict[str, TensorList] = {}
    for k,v in optimizerstate.items():
        if k.endswith('.buffer'):
            buffergroupname = '.'.join(k.split('.')[:-2])
            buffers[buffergroupname] = buffers.get(buffergroupname, []) + [v]
        else:
            params[k] = v

    return params, buffers

def _unpack_params_from_optimizerstate(optimizerstate:TensorDict) -> TensorList:
    '''Unpack a flat dict and return only parameters as a list,
       because torch.jit.trace() cannot handle or something.'''
    return list(_unpack_optimizerstate(optimizerstate)[0].values())


# I really hate this but I haven't found an easier solution yet
# torch.jit.script requires that the scripted function has annotations
# so this function takes annotations from module and writes a new module to file
def subclass_via_tempfile(module:torch.nn.Module) \
-> tp.Tuple[type[TrainStep], tempfile.TemporaryDirectory]|Exception:
    module_anns = module.forward.__annotations__
    if 'x' not in module_anns or 't' not in module_anns:
        return Exception(
            'Module must accept annotated inputs `x` and targets `t`'
        )
    x_type = type_to_str(module_anns['x'])
    t_type = type_to_str(module_anns['t'])
    trainstep_src = trainstep_src_template.format(x_type=x_type, t_type=t_type)

    tempdir  = tempfile.TemporaryDirectory()
    temppath = os.path.join(tempdir.name, 'trainstep.py')
    open(temppath, 'w').write(trainstep_src)

    uid  = str(uuid.uuid4())
    spec = importlib.util.spec_from_file_location('tempmodule'+uid, temppath)
    if spec is None or spec.loader is None:
        return Exception('Unexpected error')
    
    tempmodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tempmodule)

    return tempmodule.TrainStep, tempdir

def type_to_str(t:type) -> str:
    if t == torch.Tensor:
        return 'torch.Tensor'
    else:
        return str(t)

trainstep_src_template = '''
import torch
import typing
from traininglib.torchscriptlib import TrainStep as BaseTrainStep, TensorDict

class TrainStep(BaseTrainStep):
    def forward(self, x:{x_type}, t:{t_type}, optimizerstate:TensorDict):
        old_params = self._prepare_forward(optimizerstate)
        loss = self.module(x, t)
        output = self._backward_step(loss, optimizerstate)
        self._set_parameters(old_params)
        return output
'''


