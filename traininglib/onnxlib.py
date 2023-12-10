import typing as tp
import torch
import numpy as np

import os
import io
import json
import copy
import operator
import sys
import warnings
warnings.simplefilter('ignore', UserWarning)  #pytorch throws too many of those
import zipfile

from torch.fx.experimental.proxy_tensor import make_fx

import onnxscript
from onnxscript.onnx_opset import opset16 as op
from onnxscript import FLOAT
import onnx
opset_version = 16


from torch.optim.sgd import sgd         # type: ignore
from torch.optim.adamw import adamw     # type: ignore


StateDict  = tp.Dict[str, torch.nn.Parameter]
TensorDict = tp.Dict[str, torch.Tensor]


class ExportedInferenceONNX(tp.NamedTuple):
    #inference step in onnx format
    onnx_bytes: bytes
    #onnx inputfeed, i.e. state dict, excluding `x`
    inputfeed:  tp.Dict[str, np.ndarray]

    def save_as_zipfile(self, path:str, x:np.ndarray) -> None:
        if not path.endswith('.pt.zip'):
            path = f'{path}.pt.zip'
        base = os.path.splitext(os.path.basename(path))[0]
        with zipfile.ZipFile(path, 'w') as zipf:
            zipf.writestr(f'{base}/onnx/inference.onnx', self.onnx_bytes)
            schema = {}
            for i,(k,v) in enumerate(self.inputfeed.items()):
                vpath = f'{base}/.data/{i}.storage'
                zipf.writestr(vpath, v.tobytes())
                schema[k] = {
                    'shape': list(v.shape),
                    'dtype': str(v.dtype),
                    'path':  vpath,
                }
            schema['x'] = {
                'shape': list(x.shape),
                'dtype': str(x.dtype),
            }
            zipf.writestr(
                f'{base}/onnx/inference.schema.json', json.dumps(schema, indent=2)
            )


class DebugInfo(tp.NamedTuple):
    #fx function as converted to onnx
    train_step_tx:   tp.Callable
    #fx function with minimal modifications
    train_step_tx_unmodified:   tp.Callable


class ExportedTrainingONNX(tp.NamedTuple):
    #training steps in onnx format
    onnx_bytes:   bytes

    #onnx inputfeed (excluding `x` and `t`) for the first training step
    inputfeed:    tp.Dict[str, np.ndarray]

    #only if _debug == True
    debug: DebugInfo|None = None



def export_model_as_functional_inference_onnx(
    m: torch.nn.Module,
    x: torch.Tensor,
) -> ExportedInferenceONNX:
    m.eval()
    sd = {k:v.clone() for k,v in dict(m.state_dict()).items()}
    
    def forward_f(sd:StateDict, x:torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(m, sd, x, strict=True)
    
    fx = make_fx(forward_f, tracing_mode='fake')(sd, x)
    replace_inplace_ops(fx, bn=True)
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(fx, torch_op, manual_op)
    cleanup_graph(fx)

    inputnames  = list(sd.keys()) + ['x']
    outputnames = ['y']

    buf = io.BytesIO()
    torch.onnx.export(
        fx, 
        {'sd':sd, 'x':x},
        buf, 
        training     = torch.onnx.TrainingMode.EVAL,
        input_names  = inputnames, 
        output_names = outputnames,
        do_constant_folding = False,
        opset_version       = 16,
        #export_modules_as_functions = {
        #    type(m) for m in list(fx.modules())[1:]
        #},
    )
    onnx_bytes = buf.getvalue()
    inputnames = remove_non_inference_inputnames(inputnames)
    inputfeed  = state_dict_to_onnx_input(sd, inputnames)
    return ExportedInferenceONNX(onnx_bytes, inputfeed)


def export_model_as_functional_training_onnx(
    m:         torch.nn.Module,
    loss_func: tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x:         torch.Tensor,
    t:         torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    _debug:    bool = False,
) -> ExportedTrainingONNX:
    optim = initialize_optimizer(optimizer)
    sd = {k:v.clone() for k,v in dict(m.state_dict()).items()}
    
    def forward_step_f(
        sd_grads:    StateDict,
        sd_nongrads: StateDict,
        x:           torch.Tensor, 
        t:           torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        sd     = sd_grads | sd_nongrads
        _set_marker('submodule:forward')
        y      = torch.func.functional_call(m, sd, x, strict=True)

        _set_marker('submodule:loss')
        loss   = loss_func(y, t)
        
        _set_marker('submodule:backward')
        return loss, y

    #gradient computation function
    grad_f = torch.func.grad_and_value(forward_step_f, argnums=0, has_aux=True)

    def train_step(
        sd:      StateDict,
        x:       torch.Tensor,
        t:       torch.Tensor,
        buffers: tp.Dict[str, tp.List[torch.Tensor]],
    ):
        sd_grads, sd_nongrads = filter_nongrad_values(sd)
        gradients, (loss, y)  = grad_f(sd_grads, sd_nongrads, x, t)
        
        _set_marker('submodule:optimizer')
        new_sd_grads, buffers = run_optimizer(
            gradients, sd_grads, buffers, optim.func, optim.hyper_params
        )
        new_sd: StateDict     = new_sd_grads | sd_nongrads
        return {
            'state_dict': new_sd,
            'buffers':    buffers,
            'gradients':  gradients,
            'loss':       loss,
            'y':          y,
        }
    
    
    train_step_tx = make_fx(train_step, tracing_mode='fake')(
        sd, x, t, optim.buffers
    )
    #rename_all_nodes(train_step_tx, prefix='main')
    replace_inplace_ops(train_step_tx, bn=True)
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(train_step_tx, torch_op, manual_op)
    cleanup_graph(train_step_tx)

    if _debug:
        raise NotImplementedError('TODO')
        # train_step_tx = _return_all_nodes(train_step_tx)

        # #unmodified, i.e. no decompositions
        # train_step_tx_unmodified = make_fx(train_step, tracing_mode='fake')(
        #     sd, x, t, _init_out['buffers']
        # )
        # rename_all_nodes(train_step_tx_unmodified, prefix='main')
        # #modified after all, removing inplace ops
        # replace_inplace_ops(train_step_tx_unmodified, bn=True)
        # cleanup_graph(train_step_tx_unmodified)
        # train_step_tx_unmodified = _return_all_nodes(train_step_tx_unmodified)

    #NOTE: disabled because onnx export segfaults on large(-ish) models
    #create_submodules_from_markers(train_step_tx)

    #TODO: fake buffers
    _out = train_step_tx(sd, x, t, optim.buffers)

    buffernames:tp.List[str] = []
    for buffername, bufs in optim.buffers.items():
        buffernames += [f'{buffername}.{i}.buffer' for i,b in enumerate(bufs)]

    inputnames  = list(sd.keys()) + ['x'] + ['t'] + buffernames
    outputnames = [f'{k}.output' for k in _out['state_dict']]         \
                + [f'{k}.output' for k in buffernames]                \
                + [f'{k}.gradient.output' for k in _out['gradients']] \
                + ['loss', 'y']

    if _debug:
        raise NotImplementedError('TODO')
        # outputnames_0 += [f'{k}.debug' for k in _init_out['debug'].keys()]
        
        # _2nd_out = train_step_tx(sd, x, t, _init_out['buffers'])
        # outputnames   += [f'{k}.debug' for k in _2nd_out['debug'].keys()]


    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx, 
        {'sd':sd, 'x':x, 't':t, 'buffers':optim.buffers},
        buf, 
        training     = torch.onnx.TrainingMode.TRAINING,
        input_names  = inputnames, 
        output_names = outputnames,
        do_constant_folding = False,
        opset_version       = 16,
        #NOTE: this messes up outputnames, don't use for now
        #export_modules_as_functions = {
        #    type(m) for m in list(train_step_tx.modules())[1:]
        #},
    )
    onnx_bytes = buf.getvalue()

    inputfeed = state_dict_to_onnx_input(sd, inputnames)
    inputfeed = inputfeed | buffers_to_onnx_input(optim.buffers)

    if not _debug:
        return ExportedTrainingONNX(onnx_bytes, inputfeed)
    else:
        raise NotImplemented('TODO')
        # return ExportedONNX(
        #     onnx_bytes, inputfeed, debug = DebugInfo(
        #         train_step_tx,
        #         train_step_tx_unmodified,
        #     )
        # )


class OptimizerState(tp.NamedTuple):
    func:         tp.Callable
    hyper_params: tp.Dict[str, tp.Any]
    buffers:      tp.Dict[str, tp.List[torch.Tensor]]

def initialize_optimizer(optim:torch.optim.Optimizer) -> OptimizerState:
    assert len(optim.state) == 0, 'Optimizer must not be initialized'
    
    assert len(optim.param_groups) == 1
    group = optim.param_groups[0]
    new_params = [p.clone().detach() for p in group['params']]
    for p in new_params:
        p.requires_grad = True
        p.grad = torch.zeros_like(p)
    group = dict(group) | {'params':new_params}
    optim = optim.__class__(**group)

    if isinstance(optim, torch.optim.SGD):
        return initalize_sgd(optim)
    elif isinstance(optim, torch.optim.AdamW):
        return initialize_adamw(optim)
    else:
        raise NotImplementedError(optim.__class__)
    return True

def initalize_sgd(optim:torch.optim.SGD) -> OptimizerState:
    group = optim.param_groups[0]
    assert group['dampening'] == 0

    momentum_buffer_list: tp.List[torch.Tensor] = [
        torch.zeros_like(p) for p in group['params']
    ]
    buffers = {'momentum_buffer_list': momentum_buffer_list}
    hyper_params = {
        k:group[k] for k in [
            'lr', 'momentum', 'weight_decay', 'dampening', 'nesterov', 'maximize'
        ]
    }
    return OptimizerState(
        func         = sgd,
        hyper_params = hyper_params,
        buffers      = buffers,

    )

def initialize_adamw(optim:torch.optim.AdamW) -> OptimizerState:
    group = optim.param_groups[0]
    exp_avgs:        tp.List[torch.Tensor] = []
    exp_avg_sqs:     tp.List[torch.Tensor] = []
    max_exp_avg_sqs: tp.List[torch.Tensor] = []
    steps:           tp.List[torch.Tensor] = []
    amsgrad: bool = group['amsgrad']
    has_complex = optim._init_group( # type: ignore [attr-defined]
        group, [], [], amsgrad, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, steps
    )
    assert not has_complex

    buffers = {
        'exp_avgs':        exp_avgs,
        'exp_avg_sqs':     exp_avg_sqs,
        'max_exp_avg_sqs': max_exp_avg_sqs,
        'state_steps':     steps
    }
    hyper_params = {
        k: group[k] for k in [
            'lr', 'eps', 'amsgrad', 'weight_decay', 'maximize', 
        ]
    } | {
        'beta1': group['betas'][0], 
        'beta2': group['betas'][1],
        'differentiable': True,      # important
    }
    
    return OptimizerState(
        func         = adamw,
        hyper_params = hyper_params,
        buffers      = buffers,
    )


def run_optimizer(
    gradients:    TensorDict,
    parameters:   StateDict,
    buffers:      tp.Dict[str, tp.List[torch.Tensor]],
    func:         tp.Callable,
    hyper_params: tp.Dict[str, tp.Any],
) -> tp.Tuple[StateDict, tp.Dict[str, tp.List[torch.Tensor]]]:
    keys: tp.List[str] = list(parameters.keys())
    parameters_list: tp.List[torch.Tensor] = [parameters[k] for k in keys]
    gradients_list:  tp.List[torch.Tensor] = [gradients[k]  for k in keys]

    func(
        parameters_list,
        gradients_list,
        **buffers,
        **hyper_params,
    )
    #values are updated inplace
    new_parameters = dict(zip(keys, parameters_list))
    return new_parameters, buffers   # type: ignore [return-value]


def state_dict_to_onnx_input(sd:StateDict, onnxnames:tp.List[str]) -> tp.Dict[str, np.ndarray]:
    inputs = { k:v.data.numpy() for k,v in sd.items() }
    inputs = { k:v for k,v in inputs.items() if k in onnxnames }
    assert len(inputs)
    return inputs

def buffers_to_onnx_input(buffers:tp.Dict[str, tp.List[torch.Tensor]]) -> tp.Dict[str, np.ndarray]:
    flat:tp.Dict[str, np.ndarray] = {}
    for bufname, bufs in buffers.items():
        for i,buf in enumerate(bufs):
            flat[f'{bufname}.{i}.buffer'] = buf.numpy()
    return flat


def filter_nongrad_values(sd:StateDict) -> tp.Tuple[StateDict, StateDict]:
    '''Split a state dict into two parts, one with values that require a gradient 
       and are updated during backprop and the remaining ones'''
    #TODO: use .requires_grad
    names   = ['num_batches_tracked', 'running_mean', 'running_var']
    sd_grad = {
        k:v for k,v in sd.items() if not any([k.endswith(name) for name in names])
    }
    sd_nongrad = {
        k:v for k,v in sd.items() if k not in sd_grad
    }
    return sd_grad, sd_nongrad 

def remove_non_inference_inputnames(inputnames:tp.List[str]) -> tp.List[str]:
    to_remove = ['num_batches_tracked']
    for pattern in to_remove:
        inputnames = [n for n in inputnames if pattern not in n]
    return inputnames


def dilate(x:torch.Tensor, dilation:tp.Tuple[int,int]) -> torch.Tensor:
    dilation = tuple(dilation) # type: ignore
    assert dilation in [(1,1), (2,2)], NotImplemented
    if dilation == (1,1):
        return x
    
    B,C,H,W = x.shape
    indices = torch.arange(W)*2
    indices = indices.reshape(1,1,1,-1)
    indices = indices.expand(B,C,W,-1)
    #NOTE: not using zeros() because it would be interpreted as a constant 
    #and stored directly in onnx, blowing up the file size
    #b = torch.zeros([B,C,H,W*2])
    z = torch.cat([x,x], dim=-1) * 0.0
    b = torch.scatter(z, 3, indices, x)

    indices = torch.arange(H)*2
    indices = indices.reshape(1,1,-1,1)
    indices = indices.expand(B,C,-1,W*2)
    #NOTE: not using zeros() because it would be interpreted as a constant 
    #and stored directly in onnx, blowing up the file size
    #c = torch.zeros([B,C,H*2,W*2])
    z = torch.cat([z,z], dim=-2) * 0.0
    c = torch.scatter(z, 2, indices, b)

    return c

def manual_convolution_backward(
    grad_out:   torch.Tensor,
    input:      torch.Tensor,
    weight:     torch.Tensor,
    _:          tp.Any,       #not used
    stride:     tp.Tuple[int,int], 
    padding:    tp.Tuple[int,int], 
    dilation:   tp.Tuple[int,int], 
    transposed: bool, 
    outpadding: tp.Tuple[int,int], 
    groups:     int,
    outmask:    tp.Tuple[bool,bool,bool],
) -> tp.List[torch.Tensor|None]:
    '''Unoptimized and incomplete implementation of the convolution backward pass.
       Can be exported to ONNX'''
    B,C,H,W = input.shape

    input_T = input.transpose(0,1)
    if groups != 1:
        input_T = input_T.reshape(C//groups, B*groups, H, W)
    grad_out_T    = grad_out.transpose(0,1)
    dilation_gw   = stride
    stride_gw     = [1,1]  #maybe = dilation?
    grad_weight_T = torch.ops.aten.convolution.default(
        input_T, grad_out_T, None, stride_gw, padding, dilation_gw, transposed, outpadding, groups
    )
    grad_weight = grad_weight_T.transpose(0,1)
    #need to truncate shape, can happen on odd dimensions
    grad_weight = grad_weight[..., :weight.shape[2], :weight.shape[3]]

    grad_bias          = None
    grad_bias_required = outmask[2]
    if grad_bias_required:
        grad_bias     = grad_out.to(torch.float64).sum(dim=[0,2,3]).to(torch.float32)

    grad_input        = None
    grad_input_needed = outmask[0]
    if grad_input_needed:
        if groups != 1:
            weight = weight.reshape(C//groups, weight.shape[1]*groups, *weight.shape[2:] )
        flip_dims     = list(range(2, len(weight.shape)))
        weight_flip   = torch.flip(weight, dims=flip_dims)
        weight_flip_T = weight_flip.transpose(0,1)
        padding_gi    = list(s-p-1 for s,p in zip(weight.shape[2:], padding))
        grad_out_dil  = dilate(grad_out, dilation=stride)
        stride_gi     = [1,1]
        dilation_gi   = [1,1]
        grad_input    = torch.ops.aten.convolution.default(
            grad_out_dil, 
            weight_flip_T, 
            None, 
            stride_gi, 
            padding_gi, 
            dilation_gi, 
            transposed, 
            outpadding, 
            groups,
        )
        #need to truncate shape, can happen on odd dimensions
        grad_input = grad_input[..., :input.shape[2], :input.shape[3]]
    return [grad_input, grad_weight, grad_bias]


def manual_max_pool2d_with_indices_backward(
    grad_output: torch.Tensor,
    input:       torch.Tensor,
    kernel_size: tp.Tuple[int,int],
    stride:      tp.Tuple[int,int],
    padding:     tp.Tuple[int,int],
    dilation:    tp.Tuple[int,int],
    ceil_mode:   bool,
    indices:     torch.Tensor,
) -> torch.Tensor:
    '''Manual implementation of torch.ops.aten.max_pool2d_with_indices_backward.
       Can be exported to ONNX.'''
    intermediate_shape = input.shape[:2] + (-1,)
    #NOTE: not using zeros_like() because it would be interpreted as a constant 
    #and stored directly in onnx, blowing up the file size
    #z = torch.zeros_like(input.reshape(intermediate_shape))
    z = input.reshape(intermediate_shape) * 0.0
    grad_output = grad_output.reshape(intermediate_shape)
    indices     = indices.reshape(intermediate_shape)
    output      = torch.scatter_add(z, 2, indices, grad_output)
    return output.reshape(input.shape)


def replace_op_type_with_custom_function(
    gm:              torch.fx.graph_module.GraphModule, 
    optype:          tp.Callable, 
    custom_function: tp.Callable,
    reuse_name:      bool = True
):
    '''Register `custom_function` as a submodule in the traced graph module 
       and replace all `optype` calls with it. '''
    class custom_module(torch.nn.Module):
        def forward(self, *a, **kw):
            return custom_function(*a, **kw)
    modulename = custom_function.__name__
    custom_module.__name__ = modulename
    custom_module.__module__ = modulename

    gm.add_submodule(modulename, custom_module())
    for node in gm.graph.nodes:
        if node.target == optype:
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_module(modulename, node.args, node.kwargs)
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            if reuse_name:
                new_node.name = node.name
    gm.graph.lint()



def replace_inplace_ops(gm:torch.fx.graph_module.GraphModule, bn:bool=False):
    decompositions = {
        torch.ops.aten.add_.Tensor: torch.ops.aten.add,
        torch.ops.aten.mul_.Tensor: torch.ops.aten.mul,
        torch.ops.aten.addcdiv_.default: torch.ops.aten.addcdiv,
        torch.ops.aten.addcmul_.default: torch.ops.aten.addcmul,
    }
    if bn:
        #decompositions[torch.ops.aten.native_batch_norm.default] = functional_batch_norm
        decompositions[torch.ops.aten.native_batch_norm.default] \
            = torch.ops.aten._native_batch_norm_legit_functional.default
        decompositions[torch.ops.aten._native_batch_norm_legit.default] \
            = torch.ops.aten._native_batch_norm_legit_functional.default
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(gm, torch_op, manual_op)
    
    #replace the first (!) occurrence of running mean/var in the output list for each batchnorm
    #because not inplace anymore
    outputnode = list(gm.graph.nodes)[-1]
    outputlist = list(outputnode.args[0])
    for node in gm.graph.nodes:
        if node.target in ['functional_batch_norm', 'functional_batch_norm_64','_native_batch_norm_legit_functional.default']:
            with gm.graph.inserting_after(node):
                old_running_mean  = node.args[3]
                old_running_var   = node.args[4]
                new_running_mean  = gm.graph.call_function(operator.getitem, (node, 3))
                new_running_var   = gm.graph.call_function(operator.getitem, (node, 4))
            
            if old_running_mean in outputlist:
                outputlist[outputlist.index(old_running_mean)] = new_running_mean
            if old_running_var in outputlist:
                outputlist[outputlist.index(old_running_var)] = new_running_var
    
    with gm.graph.inserting_after(outputnode):
        new_outputnode = gm.graph.output(outputlist)
    outputnode.replace_all_uses_with(new_outputnode)
    gm.graph.erase_node(outputnode)
    gm.graph.lint()


def batch_norm_functional_no_training(
    x, weight, bias, running_mean, running_var, momentum, eps,
):
    training   = False
    functional = True
    return torch._decomp.decompositions.native_batch_norm_helper(
        x, weight, bias, running_mean, running_var, training, momentum, eps, functional
    )


@onnxscript.script() # type: ignore
def batch_norm_functional_training(
    x:           FLOAT, 
    weight:      FLOAT, 
    bias:        FLOAT, 
    running_mean:FLOAT, 
    running_var: FLOAT, 
    #training:   bool,  #ignored, implicitly True 
    momentum:    float, 
    eps:         float,
    N:           float,  # x.shape[0] * x.shape[2] * x.shape[3], fixed shape
):
    # cast to float64, pytorch seems to do it too internally
    x = op.Cast(x, to=onnx.TensorProto.DOUBLE) # type: ignore [assignment]

    reduction_axes = [0,2,3]
    x_mean         = op.ReduceMean(x, axes=[0,2,3])  #NOTE using reduction_axes gives error
    x_norm         = (x - x_mean) # type: ignore
    x_norm_squared = x_norm * x_norm
    x_norm_sqsum   = op.ReduceSum(x_norm_squared, axes=reduction_axes) # type: ignore
    x_biased_var   = x_norm_sqsum / N
    x_biased_std   = op.Sqrt(x_biased_var + eps)
    rstd           = op.Div(1.0, x_biased_std)

    #like pytorch but numerically less stable
    #output = (x - x_mean) * rstd
    
    #more stable (i think, actually no difference)
    output = x_norm / x_biased_std
    
    x_mean       = op.Cast(x_mean,       to=onnx.TensorProto.FLOAT) # type: ignore
    x_norm_sqsum = op.Cast(x_norm_sqsum, to=onnx.TensorProto.FLOAT)
    output       = op.Cast(output,       to=onnx.TensorProto.FLOAT)
    rstd         = op.Cast(rstd,         to=onnx.TensorProto.FLOAT)

    weight = op.Unsqueeze(weight, axes=[-2,-1]) # type: ignore
    bias   = op.Unsqueeze(bias,   axes=[-2,-1]) # type: ignore
    output = output * weight + bias

    save_mean = op.Squeeze(x_mean, axes=reduction_axes)  # type: ignore
    save_rstd = op.Squeeze(rstd,   axes=reduction_axes)  # type: ignore

    N_unbiased       = N - 1.0
    unbiased_var     = x_norm_sqsum / N_unbiased
    unbiased_var_sqz = op.Squeeze(unbiased_var, axes=reduction_axes) # type: ignore
    inv_momentum     = 1.0 - momentum
    new_running_mean = momentum * save_mean        + inv_momentum * running_mean # type: ignore
    new_running_var  = momentum * unbiased_var_sqz + inv_momentum * running_var  # type: ignore

    return output, save_mean, save_rstd, new_running_mean, new_running_var




GraphContext = torch.onnx._internal.jit_utils.GraphContext

def onnx_squeeze(g:GraphContext, x:torch.Value, dim:torch.Value|None = None):
    if dim is None:
        return g.op('Squeeze', x)
    
    dims_to_squeeze = dim.node().t('value').long()
    if dims_to_squeeze.ndim == 0:
        dims_to_squeeze = dims_to_squeeze[None]
    return g.op("Squeeze", x, g.op("Constant", value_t=dims_to_squeeze))

torch.onnx.register_custom_op_symbolic('::squeeze', onnx_squeeze, opset_version=16)


@onnxscript.script() # type: ignore [arg-type]
def onnx_sign(X):
    return op.Sign(X)

def aten_sgn(g:GraphContext, X:torch.Value):
    return g.onnxscript_op(onnx_sign, X).setType(X.type())

torch.onnx.register_custom_op_symbolic('aten::sgn', aten_sgn, opset_version=16)


@onnxscript.script() # type: ignore [arg-type]
def onnx_threshold(grad_output, self, threshold):
    threshold_casted = op.CastLike(threshold, self)
    mask = self <= threshold_casted
    zero = op.CastLike(0, grad_output) # type: ignore [type-var]
    return op.Where( mask, zero, grad_output )

def aten_threshold_backward(
    g:GraphContext, grad_output:torch.Value, self:torch.Value, threshold:torch.Value
) -> torch.Value:
    return g.onnxscript_op(
        onnx_threshold, grad_output, self, threshold
    ).setType(grad_output.type())

torch.onnx.register_custom_op_symbolic(
    'aten::threshold_backward', aten_threshold_backward, opset_version=16
)


def aten_where(g:GraphContext, mask, x0, x1):
    if x0.type().dim() < x1.type().dim():
        x0 = g.op("CastLike", x0, x1)
    else:
        x1 = g.op("CastLike", x1, x0)
    return g.op('Where', mask, x0, x1)

torch.onnx.register_custom_op_symbolic(
    'aten::where', aten_where, opset_version=16
)


def aten_addcdiv(
    g:GraphContext, 
    input:   torch.Value, 
    tensor1: torch.Value, 
    tensor2: torch.Value,
    value:   torch.Value,
):
    ratio = g.op('Div', tensor1, tensor2)
    mul   = g.op('Mul', ratio, value)
    add   = g.op('Add', mul, input)
    return add

torch.onnx.register_custom_op_symbolic(
    'aten::addcdiv', aten_addcdiv, opset_version=16
)



def aten_bnfunc(
    g:        GraphContext, 
    x:        torch.Value, 
    w:        torch.Value, 
    b:        torch.Value, 
    rm:       torch.Value, 
    rv:       torch.Value, 
    training: torch.Value,
    momentum: torch.Value,
    eps:      torch.Value,
):
    momentum_value = float(momentum.node().t('value'))
    eps_value      = float(eps.node().t('value'))
    x_sizes        = x.type().sizes()  # type: ignore [attr-defined]
    N              = int(x_sizes[0] * x_sizes[2] * x_sizes[3])
    y, sm,sv, nrm,nrv = g.onnxscript_op(
        batch_norm_functional_training, 
        x, 
        w, 
        b, 
        rm, 
        rv, 
        momentum_f = momentum_value, 
        eps_f      = eps_value,
        N_f        = float(N),
        outputs    = 5, 
    )
    y.setType(x.type())
    sm.setType(rm.type())
    sv.setType(rv.type())
    nrm.setType(rm.type())
    nrv.setType(rv.type())
    return y, sm,sv, nrm,nrv

torch.onnx.register_custom_op_symbolic(
    'aten::_native_batch_norm_legit_functional', aten_bnfunc, opset_version=16
)


@onnxscript.script() # type: ignore [arg-type]
def aten_upsample_nearest2d_backward_with_constants(
    grad_out, outputsize, inputsize, indices0, repeats0, indices1, repeats1
):
    z = op.ConstantOfShape(outputsize) # value implicitly zero

    indices0 = op.Tile(indices0, repeats0)
    z0 = op.ScatterElements(z, indices0, grad_out, axis=3, reduction='add')

    indices1 = op.Tile(indices1, repeats1)
    z1 = op.ScatterElements(z, indices1, z0, axis=2, reduction='add')

    sliced = op.Slice(z1, starts=[0,0,0,0], ends=inputsize)

    return sliced


def onnx_upsample_nearest2d_backward(
    g:GraphContext, 
    grad_out:torch.Value,
    outputsize:torch.Value,
    inputsize:torch.Value,
    scales_h:torch.Value,
    scales_w:torch.Value,
):
    assert isinstance(scales_h.type(), torch.NoneType), NotImplementedError()
    assert isinstance(scales_w.type(), torch.NoneType), NotImplementedError()
    outputsize_t = outputsize.node().t('value')
    inputsize_t  = inputsize.node().t('value')

    outputsize_full = torch.cat([inputsize_t[:2], outputsize_t])
    outputsize_full_op = g.op('Constant', value_t=outputsize_full)

    ixs0 = np.arange(
        0, inputsize_t[-1], inputsize_t[-1]/outputsize_t[-1]
    ).astype(int)[None,None,None]
    ixs0_const = g.op('Constant', value_t=torch.as_tensor(ixs0))
    repeats0  = list(inputsize_t)[:2]+[outputsize_t[-2], 1]
    repeats0_const = g.op('Constant', value_t=torch.as_tensor(repeats0))

    ixs1 = np.arange(
        0, inputsize_t[-2], inputsize_t[-2]/outputsize_t[-2]
    ).astype(int)[None,None,:,None]
    ixs1_const = g.op('Constant', value_t=torch.as_tensor(ixs1))
    repeats1  = list(inputsize_t)[:2]+[1,outputsize_t[-1]]
    repeats1_const = g.op('Constant', value_t=torch.as_tensor(repeats1))

    output_type = grad_out.type().with_sizes(list(inputsize_t))
    return g.onnxscript_op(
        aten_upsample_nearest2d_backward_with_constants, 
        grad_out, 
        outputsize_full_op, 
        inputsize, 
        ixs0_const, 
        repeats0_const, 
        ixs1_const, 
        repeats1_const
    ).setType(output_type)

torch.onnx.register_custom_op_symbolic(
    'aten::upsample_nearest2d_backward', 
    onnx_upsample_nearest2d_backward, 
    opset_version=16
)


def cleanup_graph(gm:torch.fx.graph_module.GraphModule):
    '''Remove torch.ops.aten.clone, and torch.ops.aten.detach ops'''
    for node in gm.graph.nodes:
        if node.target in [torch.ops.aten.clone.default, torch.ops.aten.detach.default]:
            assert len(node.args) == 1
            inputnode = node.args[0]
            node.replace_all_uses_with(inputnode)
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()


def replace_all_squeeze_dims(gm:torch.fx.graph_module.GraphModule):
    graph:torch.fx.graph.Graph = gm.graph
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.squeeze.dims:
            with graph.inserting_before(node):
                tensor, dims = node.args
                shape = tensor.meta['tensor_meta'].shape
                new_shape = [shape[i] for i in range(len(shape)) if shape[i] > 1 or i not in dims ]
                #new_squeeze = graph.call_function(torch.ops.aten.squeeze, (node.args[0], ))
                new_squeeze = graph.call_function(torch.reshape, (tensor, new_shape))
                new_squeeze.name = node.name
                node.replace_all_uses_with(new_squeeze)
            graph.erase_node(node)
    graph.lint()

def rename_all_nodes(gm:torch.fx.graph_module.GraphModule, prefix:str):
    for node in gm.graph.nodes:
        node.name = f'{prefix}.{node.name}'



def _set_marker(name:str) -> None:
    '''Create a constant value that will be picked up by `make_fx()` 
       and used for post-processing the fx graph later '''
    torch.ByteTensor(list(name.encode('utf8')))


def rename_all_nodes_from_markers(gm:torch.fx.graph_module.GraphModule):
    '''Rename all graph nodes by adding a prefix previously set by _set_marker().
       Additionally, remove the markers.'''
    current_marker:str|None = None
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.lift_fresh_copy.default:
            constant = node.args[0]
            marker_as_tensor = getattr(gm, constant.target)
            current_marker = marker_as_tensor.numpy().tobytes().decode('utf8')
            gm.graph.erase_node(node)
            gm.graph.erase_node(constant)
        elif current_marker is not None:
            node.name = f'{current_marker}_{node.name}'
    gm.graph.lint()

def create_submodules_from_markers(gm:torch.fx.graph_module.GraphModule):
    '''Move graph nodes to functions according to markers set by _set_marker().
       Also remove the markers.'''
    current_marker:str|None = None
    marked_nodes:tp.Dict[str, tp.List[torch.fx.Node]] = {}

    for node in gm.graph.nodes:
        if node.target == 'output':
            continue
            
        if node.target == torch.ops.aten.lift_fresh_copy.default:
            constant = node.args[0]
            marker_as_tensor = getattr(gm, constant.target)
            current_marker = marker_as_tensor.numpy().tobytes().decode('utf8')
            gm.graph.erase_node(node)
            gm.graph.erase_node(constant)
            continue
        
        if current_marker is not None:
            marked_nodes[current_marker] \
                = marked_nodes.get(current_marker, []) + [node]
    
    gm.graph.lint()
    gm.recompile()
    for marker, nodes in marked_nodes.items():
        nodes = [n for n in nodes if n in gm.graph.nodes]
        move_nodes_to_submodule(gm, nodes, marker)
    gm.graph.lint()
    gm.recompile()

def filter_nodes_with_external_users(
    nodes:tp.List[torch.fx.Node]
) -> tp.List[torch.fx.Node]:
    '''Return nodes that have users that are not in `nodes`'''
    nodes_with_external_users:tp.List[torch.fx.Node] = []
    for n in nodes:
        for user in n.users:
            if user not in nodes:
                nodes_with_external_users.append(n)
                break
    return nodes_with_external_users

def move_nodes_to_submodule(
    gm:torch.fx.graph_module.GraphModule, nodes:tp.List[torch.fx.Node], name:str
):
    '''Extract a list of nodes from graph and move them into an own submodule'''
    new_graph = torch.fx.Graph()
    #mapping inputs of new_graph to nodes in gm
    new_graph_inputs:tp.Dict[torch.fx.Node, torch.fx.Node] = {}
    #mapping nodes in gm to nodes in new_graph
    val_remap:tp.Dict[torch.fx.Node, torch.fx.Node] = {}
    
    def remap_or_placeholder(n:torch.fx.Node) -> torch.fx.Node:
        if n in val_remap:
            return val_remap[n]
        placeholder  = new_graph.placeholder(n.name)
        val_remap[n] = placeholder
        new_graph_inputs[placeholder] = n
        return placeholder
    
    #copy nodes into new graph
    for n in nodes:
        new_n = new_graph.node_copy(n, arg_transform=remap_or_placeholder)
        val_remap[n] = new_n
    #create an output node in new graph
    old_output_nodes = filter_nodes_with_external_users(nodes)
    new_output_nodes = [val_remap[n] for n in old_output_nodes]
    new_graph.output(new_output_nodes)

    #add submodule into old graph
    submodule = torch.fx.graph_module.GraphModule(gm, new_graph, name)
    submodule.__class__.__module__ = name
    gm.add_submodule(name, submodule)
    with gm.graph.inserting_before(nodes[0]):
        #call submodule
        call_node = gm.graph.call_module(name, tuple(new_graph_inputs.values()))
        #replace output nodes in old graph
        for i, old_out_n in enumerate(old_output_nodes):
            new_out_n = gm.graph.call_function(operator.getitem, (call_node, i))
            old_out_n.replace_all_uses_with(new_out_n)

    #remove old nodes
    for n in nodes[::-1]:
        gm.graph.erase_node(n)
    
    gm.graph.lint()
    return gm


def _return_all_nodes(gm:torch.fx.graph_module.GraphModule) -> torch.fx.graph_module.GraphModule:
    '''Modify the return statement to additionally return all nodes. 
       Input graph module must already return a dict.  (For debugging)'''
    gm = copy.deepcopy(gm)

    all_nodes = list(gm.graph.nodes)
    last_node = all_nodes[-1]
    all_but_last_nodes = all_nodes[:-1]
    #nodes that return tuples currently dont work
    all_but_last_no_tuples = []  # type: ignore
    for n in all_but_last_nodes:
        for user in n.users:
            if user.target == operator.getitem:
                break
        else:
            all_but_last_no_tuples.append(n)
    #also filter out unused nodes (discarded by onnx and lead to issues)
    all_but_last_no_tuples = [n for n in all_but_last_no_tuples if len(n.users) > 0]

    debug_spec = torch.utils._pytree.TreeSpec(
        type           = dict, 
        context        = [str(n) for n in all_but_last_no_tuples], 
        children_specs = [torch.utils._pytree.LeafSpec() for n in all_but_last_no_tuples]
    )
    old_info     = gm.graph._codegen.pytree_info  # type: ignore
    old_out_spec = old_info.out_spec
    new_out_spec = torch.utils._pytree.TreeSpec(
        type           = dict,
        context        = old_out_spec.context + ['debug'],
        children_specs = old_out_spec.children_specs + [debug_spec],
    )

    gm._out_spec = new_out_spec
    gm.graph._codegen.pytree_info = torch.fx.graph._PyTreeInfo( # type: ignore
        old_info.orig_args, old_info.in_spec, new_out_spec
    )

    with gm.graph.inserting_before(last_node):
        new_node = gm.graph.output(last_node.args[0] + all_but_last_no_tuples)
    gm.graph.erase_node(last_node)
    gm.graph.lint()
    gm.recompile()
    return gm



decompositions = {
    torch.ops.aten.convolution_backward.default: manual_convolution_backward,
    torch.ops.aten._native_batch_norm_legit_no_training.default:
        batch_norm_functional_no_training,
    torch.ops.aten.max_pool2d_with_indices_backward.default: 
        manual_max_pool2d_with_indices_backward,
    torch.ops.aten.native_batch_norm_backward.default: 
        torch._decomp.decompositions.native_batch_norm_backward,
    torch.ops.aten.nll_loss_forward.default: 
        torch._decomp.decompositions.nll_loss_forward,
    torch.ops.aten.nll_loss_backward.default: 
        torch._decomp.decompositions.nll_loss_backward,
    torch.ops.aten._log_softmax_backward_data.default:
        torch._decomp.decompositions._log_softmax_backward_data,
    #inplace
    torch.ops.aten.hardswish_.default: torch.ops.aten.hardswish,
    torch.ops.aten.hardswish_backward.default:
        torch._decomp.decompositions.hardswish_backward,
    torch.ops.aten.hardsigmoid_backward.default:
        torch._decomp.decompositions.hardsigmoid_backward,
    torch.ops.aten.lerp_.Scalar:
        torch.ops.aten.lerp.Scalar,
}
