import typing as tp
import torch
import numpy as np
import io
import copy
import operator
import warnings
warnings.simplefilter('ignore', UserWarning)  #pytorch throws too many of those

from torch.fx.experimental.proxy_tensor import make_fx

import sys
#import torch.optim.sgd does not work
sgd_module = sys.modules['torch.optim.sgd']


StateDict  = tp.Dict[str, torch.nn.Parameter]
TensorDict = tp.Dict[str, torch.Tensor]


class DebugInfo(tp.NamedTuple):
    #fx functions as converted to onnx
    train_step_tx_0: tp.Callable
    train_step_tx:   tp.Callable
    #fx functions without modifications
    train_step_tx_0_unmodified: tp.Callable
    train_step_tx_unmodified:   tp.Callable


class ExportedONNX(tp.NamedTuple):
    onnx_bytes_0: bytes
    onnx_bytes:   bytes

    #only if _debug == True
    debug: DebugInfo|None = None



def export_model_as_functional_training_onnx(
    m:         torch.nn.Module,
    loss_func: tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x:         torch.Tensor,
    t:         torch.Tensor,
    _debug:    bool = False,
) -> ExportedONNX:
    sd = {k:v.clone() for k,v in dict(m.state_dict()).items()}
    
    def forward_step_f(
        sd_grads:    StateDict,
        sd_nongrads: StateDict,
        x:           torch.Tensor, 
        t:           torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        sd     = sd_grads | sd_nongrads
        y      = torch.func.functional_call(m, sd, x, strict=True)
        loss   = loss_func(y, t)
        return loss, y

    #gradient computation function
    grad_f = torch.func.grad_and_value(forward_step_f, argnums=0, has_aux=True)

    def train_step_0(
        sd: StateDict,
        x:  torch.Tensor,
        t:  torch.Tensor,
    ) -> tp.Dict[str, torch.Tensor|TensorDict|StateDict]:
        sd_grads, sd_nongrads = filter_nongrad_values(sd)
        gradients, (loss, y)  = grad_f(sd_grads, sd_nongrads, x, t)
        new_sd_grads, buffers = run_optimizer_init(gradients, sd_grads)
        new_sd: StateDict     = new_sd_grads | sd_nongrads
        return {
            'state_dict': new_sd,
            'buffers':    buffers,
            'gradients':  gradients,
            'loss':       loss,
            'y':          y,
        }

    def train_step(
        sd:      StateDict,
        x:       torch.Tensor,
        t:       torch.Tensor,
        buffers: TensorDict,
    ):
        sd_grads, sd_nongrads = filter_nongrad_values(sd)
        gradients, (loss, y)  = grad_f(sd_grads, sd_nongrads, x, t)
        new_sd_grads, buffers = run_optimizer(gradients, sd_grads, buffers) # type: ignore
        new_sd: StateDict     = new_sd_grads | sd_nongrads
        return {
            'state_dict': new_sd,
            'buffers':    buffers,
            'gradients':  gradients,
            'loss':       loss,
            'y':          y,
        }
    
    decompositions = {
        #torch.ops.aten.native_batch_norm.default: functional_aten_batch_norm,
        torch.ops.aten.native_batch_norm.default:manual_batch_norm,
        torch.ops.aten.convolution_backward.default: manual_convolution_backward,
        torch.ops.aten.max_pool2d_with_indices_backward.default: 
            manual_max_pool2d_with_indices_backward,
        torch.ops.aten.threshold_backward.default: threshold_backward,
        torch.ops.aten.native_batch_norm_backward.default: 
            torch._decomp.decompositions.native_batch_norm_backward,
        torch.ops.aten.var_mean.correction:var_mean_64,
        torch.ops.aten.rsqrt.default: rsqrt_64,
        # torch.ops.aten.nll_loss_forward.default: 
        #     torch._decomp.decompositions.nll_loss_forward,
        # torch.ops.aten.nll_loss_backward.default: 
        #     torch._decomp.decompositions.nll_loss_backward,
        torch.ops.aten.nll_loss_forward.default:  _nll_loss_forward,
        torch.ops.aten.nll_loss_backward.default: _nll_loss_backward,
        torch.ops.aten._log_softmax_backward_data.default:
            torch._decomp.decompositions._log_softmax_backward_data,
    }
    train_step_tx_0 = make_fx(train_step_0, tracing_mode='fake')(sd, x, t)
    rename_all_nodes(train_step_tx_0, prefix='main')
    replace_all_aten_sgn(train_step_tx_0)
    replace_inplace_ops(train_step_tx_0, bn=False)
    replace_all_aten_native_batch_norm(train_step_tx_0)
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(train_step_tx_0, torch_op, manual_op)
    cleanup_graph(train_step_tx_0)

    if _debug:
        train_step_tx_0 = _return_all_nodes(train_step_tx_0)

        #unmodified, i.e. no decompositions
        train_step_tx_0_unmodified = make_fx(train_step_0, tracing_mode='fake')(sd, x, t)
        rename_all_nodes(train_step_tx_0_unmodified, prefix='main')
        #modified after all, removing inplace ops
        replace_inplace_ops(train_step_tx_0_unmodified, bn=True)
        cleanup_graph(train_step_tx_0_unmodified)
        train_step_tx_0_unmodified = _return_all_nodes(train_step_tx_0_unmodified)


    _init_out = train_step_tx_0(sd, x, t)
    train_step_tx   = make_fx(train_step, tracing_mode='fake')(
        sd, x, t, _init_out['buffers']
    )
    rename_all_nodes(train_step_tx, prefix='main')
    replace_all_aten_sgn(train_step_tx)
    replace_inplace_ops(train_step_tx, bn=False)
    replace_all_aten_native_batch_norm(train_step_tx)
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(train_step_tx, torch_op, manual_op)
    cleanup_graph(train_step_tx)

    if _debug:
        train_step_tx = _return_all_nodes(train_step_tx)

        #unmodified, i.e. no decompositions
        train_step_tx_unmodified = make_fx(train_step, tracing_mode='fake')(
            sd, x, t, _init_out['buffers']
        )
        rename_all_nodes(train_step_tx_unmodified, prefix='main')
        #modified after all, removing inplace ops
        replace_inplace_ops(train_step_tx_unmodified, bn=True)
        cleanup_graph(train_step_tx_unmodified)
        train_step_tx_unmodified = _return_all_nodes(train_step_tx_unmodified)


    inputnames_0  = list(sd.keys()) + ['x'] + ['t']
    inputnames    = inputnames_0 + [f'{k}.buffer' for k in _init_out['buffers'].keys()]
    outputnames_0 = [f'{k}.output' for k in _init_out['state_dict'].keys()]         \
                  + [f'{k}.buffer.output' for k in _init_out['buffers'].keys()]     \
                  + [f'{k}.gradient.output' for k in _init_out['gradients'].keys()] \
                  + ['loss', 'y']
    outputnames   = list(outputnames_0)
    if _debug:
        outputnames_0 += [f'{k}.debug' for k in _init_out['debug'].keys()]
        
        _2nd_out = train_step_tx(sd, x, t, _init_out['buffers'])
        outputnames   += [f'{k}.debug' for k in _2nd_out['debug'].keys()]


    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx_0, 
        {'sd':sd, 'x':x, 't':t},
        buf, 
        training     = torch.onnx.TrainingMode.TRAINING,
        input_names  = inputnames_0, 
        output_names = outputnames_0,
        do_constant_folding = False,
        opset_version       = 16,
    )
    onnx_bytes_0 = buf.getvalue()

    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx, 
        {'sd':sd, 'x':x, 't':t, 'buffers':_init_out['buffers']},
        buf, 
        training     = torch.onnx.TrainingMode.TRAINING,
        input_names  = inputnames, 
        output_names = outputnames,
        do_constant_folding = False,
        opset_version       = 16,
    )
    onnx_bytes = buf.getvalue()

    if not _debug:
        return ExportedONNX(onnx_bytes_0, onnx_bytes)
    else:
        return ExportedONNX(
            onnx_bytes_0, onnx_bytes, debug = DebugInfo(
                train_step_tx_0,
                train_step_tx,
                train_step_tx_0_unmodified, 
                train_step_tx_unmodified,
            )
        )



def run_optimizer(
    gradients:  TensorDict,
    parameters: StateDict,
    buffers:    tp.Dict[str, torch.Optional[torch.Tensor]],
) -> tp.Tuple[StateDict, TensorDict]:
    keys: tp.List[str] = list(parameters.keys())
    parameters_list: tp.List[torch.Tensor] = [parameters[k] for k in keys]
    gradients_list:  tp.List[torch.Tensor] = [gradients[k]  for k in keys]
    buffers_list:    tp.List[torch.Tensor|None] = [buffers[k] for k in keys]
    sgd_module.sgd(
        parameters_list, 
        gradients_list, 
        buffers_list, 
        weight_decay = 1e-4, 
        momentum     = 0.9, 
        lr           = 0.05, 
        dampening    = 0.0, 
        nesterov     = False, 
        maximize     = False,
    )
    #values are updated inplace
    new_parameters = dict(zip(keys, parameters_list))
    new_buffers    = dict(zip(keys, buffers_list))
    return new_parameters, new_buffers  # type: ignore [return-value]

def run_optimizer_init(
    gradients:  TensorDict,
    parameters: StateDict,
) -> tp.Tuple[StateDict, TensorDict]:
    buffers: tp.Dict[str, tp.Optional[torch.Tensor]] = {k:None for k in parameters.keys()}
    return run_optimizer(gradients, parameters, buffers)


def state_dict_to_onnx_input(sd:StateDict, onnxnames:tp.List[str]) -> tp.Dict[str, np.ndarray]:
    inputs = { k:v.data.numpy() for k,v in sd.items() }
    inputs = { k:v for k,v in inputs.items() if k in onnxnames }
    assert len(inputs)
    return inputs


def filter_nongrad_values(sd:StateDict) -> tp.Tuple[StateDict, StateDict]:
    '''Split a state dict into two parts, one with values that require a gradient 
       and are updated during backprop and the remaining ones'''
    names   = ['num_batches_tracked', 'running_mean', 'running_var']
    sd_grad = {
        k:v for k,v in sd.items() if not any([k.endswith(name) for name in names])
    }
    sd_nongrad = {
        k:v for k,v in sd.items() if k not in sd_grad
    }
    return sd_grad, sd_nongrad 


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
            weight = weight.reshape(B//groups, C*groups, *weight.shape[2:] )
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


def threshold_backward(grad_output:torch.Tensor, self:torch.Tensor, threshold:float):
    #https://github.com/pytorch/pytorch/blob/fe5d8850e27d98439166c76ccc5e167fd3960df8/torch/_decomp/decompositions.py#L211C1-L212C58
    #equivalent to the link above, except that using torch.tensor(0.0, dtype=torch.float32)
    #onnx complained
    return torch.where(self <= threshold, torch.tensor(0.0, dtype=torch.float32), grad_output)

def var_mean_64(x:torch.Tensor, *a, **kw) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    '''Explicitly convert input to float64. Torch does it internally, ONNX doesnt.'''
    x = x.to(torch.float64)
    var, mean = torch.var_mean(x, *a, **kw)
    var, mean = var.to(torch.float32), mean.to(torch.float32)
    return var, mean

def rsqrt_64(x:torch.Tensor, *a, **kw) -> torch.Tensor:
    x = x.to(torch.float64)
    x = torch.ops.aten.rsqrt.default(x, *a, **kw)
    x = x.to(torch.float32)
    return x

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
    gm.recompile()


def replace_all_aten_sgn(gm:torch.fx.graph_module.GraphModule):
    '''Replace all torch.ops.aten.sgn with torch.sign, 
       because onnx cannot handle it for some reason
    '''
    graph:torch.fx.graph.Graph = gm.graph
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.sgn.default:
            with graph.inserting_before(node):
                sign = graph.call_function(torch.sign, node.args)
                node.replace_all_uses_with(sign)
            graph.erase_node(node)
    graph.lint()
    gm.recompile()



def find_node_in_graph(graph:torch.fx.graph.Graph, name:str) -> torch.fx.graph.Node|None:
    for node in graph.nodes:
        if node.name == name:
            return node
    return None

def replace_all_aten_native_batch_norm(gm:torch.fx.graph_module.GraphModule):
    '''Replace all aten_native_batch_norm nodes in a traced fx graph with aten_batch_norm'''
    graph:torch.fx.graph.Graph = gm.graph
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.native_batch_norm.default:
            replace_node_with_aten_native_batch_norm(gm.graph, node)
    graph.lint()
    gm.recompile()

    #replace_all_squeeze_dims(gm)

def replace_node_with_aten_native_batch_norm(
    graph:torch.fx.graph.Graph, node:torch.fx.graph.Node
):
    #https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1402
    empty_args = []
    for arg in node.args[:5]:
        meta = arg.meta['tensor_meta'] # type: ignore
        empty_args.append(
            torch.empty(meta.shape, dtype=meta.dtype)
        )
    new_args = empty_args + list(node.args[5:])
    new_args += [True]  #functional = True
    fx = make_fx(
        #torch._decomp.decompositions.native_batch_norm_helper, tracing_mode='fake'
        manual_batch_norm, tracing_mode='fake'
    )(*new_args)
    fx.graph.erase_node(
        find_node_in_graph(fx.graph, 'functional_1')
    )
    rename_all_nodes(fx, prefix='bn')

    node_map = { fxnode:gnode for fxnode,gnode in zip(fx.graph.nodes, node.args)}
    with graph.inserting_before(node):
        out = graph.graph_copy(fx.graph, node_map, return_output_node=True)
        fx_outputs  = list(fx.graph.nodes)[-1].args[0]
        new_outputs = tuple(node_map[n] for n in fx_outputs)
        tuple_op = graph.call_function(tuple, args=(new_outputs,))
        node.replace_all_uses_with(tuple_op)

        running_mean_node = node.args[3]
        running_var_node  = node.args[4]
        new_running_mean  = graph.call_function(operator.getitem, (tuple_op, 3))
        new_running_var   = graph.call_function(operator.getitem, (tuple_op, 4))

    outputnode = list(graph.nodes)[-1]
    with graph.inserting_before(outputnode):
        outputlist = list(outputnode.args[0])
        if running_mean_node in outputlist:
            outputlist[outputlist.index(running_mean_node)] = new_running_mean
        if running_var_node in outputlist:
            outputlist[outputlist.index(running_var_node)] = new_running_var
        new_outputnode = graph.output(outputlist)
        outputnode.replace_all_uses_with(new_outputnode)
        graph.erase_node(outputnode)
    graph.erase_node(node)
    graph.lint()
    graph.owning_module.recompile()

def functional_batch_norm(*a, **kw):
    kw = kw | {'functional': True}
    return torch._decomp.decompositions.native_batch_norm_helper(*a, **kw)

def functional_aten_batch_norm(x, w, b, rm, rv, training, momentum, eps):
    rmclone = rm *1.0 #.clone()
    rvclone = rv *1.0 #.clone()
    y = torch.ops.aten.batch_norm(
        x, w, b, rmclone, rvclone, training, momentum, eps, cudnn_enabled=False
    )
    
    #https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1402
    reduction_dims = [0] + list(range(2, x.dim()))
    biased_var, mean = torch.var_mean(
        x.to(torch.float64), dim=reduction_dims, correction=0, keepdim=True
    )
    rstd = torch.rsqrt(biased_var + eps)
    save_mean = torch.squeeze(mean, reduction_dims)
    save_rstd = torch.squeeze(rstd, reduction_dims)
    return (y, save_mean, save_rstd, rmclone, rvclone)


def functional_batch_norm_64(
    x:torch.Tensor, w:torch.Tensor, b:torch.Tensor, rm:torch.Tensor, rv:torch.Tensor, *a, **kw
):
    torch.ops.aten.batch_norm()
    kw = kw | {'functional': True}
    x  = x.to(torch.float64)
    w  = w.to(torch.float64)
    b  = b.to(torch.float64)
    rm = rm.to(torch.float64)
    rv = rv.to(torch.float64)
    outputs = torch._decomp.decompositions.native_batch_norm_helper(x, w, b, rm, rv, *a, **kw)
    outputs = [o.to(torch.float32) for o in outputs] # type: ignore
    return outputs


def manual_batch_norm(
    input:        torch.Tensor,
    weight:       torch.Tensor,
    bias:         torch.Tensor,
    running_mean: torch.Tensor,
    running_var:  torch.Tensor,
    training:     bool,
    momentum:     float,
    eps:          float,
    functional:   bool,
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #https://github.com/pytorch/pytorch/blob/dadca7aeec7caac385bafe13cc2d2d434517eb4b/torch/_decomp/decompositions.py#L1591
    #modifications: not using rsqrt to compute output for more numerical stability
    assert training
    assert functional
    reduction_dims    = [0] + list(range(2, input.dim()))
    computation_dtype = torch.float32

    input_acc = input.to(dtype=computation_dtype)
    biased_var, mean = torch.var_mean(
        input_acc, dim=reduction_dims, correction=0, keepdim=True
    )
    biased_var_plus_eps = biased_var + eps
    rstd = torch.rsqrt(biased_var_plus_eps)
    std  = torch.sqrt(biased_var_plus_eps)

    output = (input - mean) / std

    save_mean = torch.squeeze(mean, reduction_dims)
    save_rstd = torch.squeeze(rstd, reduction_dims)
    if running_mean is not None:
        new_running_mean = momentum * save_mean + (1 - momentum) * running_mean
    if running_var is not None:
        n = input.numel() / input.shape[1]
        squeezed_var = torch.squeeze(biased_var, reduction_dims)
        unbiased_var = squeezed_var * (n / (n - 1))
        new_running_var = momentum * unbiased_var + (1 - momentum) * running_var
    
    if weight is not None:
        weight = weight.flatten()
        weight = _unsqueeze_to_dim(weight, input.dim() - 1)
        output = output * weight

    if bias is not None:
        bias = bias.flatten()
        bias = _unsqueeze_to_dim(bias, input.dim() - 1)
        output = output + bias
    
    return (
        output.to(dtype=input.dtype),
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    )


def _unsqueeze_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    #https://github.com/pytorch/pytorch/blob/dadca7aeec7caac385bafe13cc2d2d434517eb4b/torch/_decomp/decompositions.py#L95
    for _ in range(dim - x.dim()):
        x = x.unsqueeze(-1)
    return x
            





def replace_inplace_ops(gm:torch.fx.graph_module.GraphModule, bn:bool=False):
    decompositions = {
        torch.ops.aten.add_.Tensor: torch.ops.aten.add,
        torch.ops.aten.mul_.Tensor: torch.ops.aten.mul,
    }
    if bn:
        decompositions[torch.ops.aten.native_batch_norm.default] = functional_batch_norm
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(gm, torch_op, manual_op)
    
    #replace the first (!) occurrence of running mean/var in the output list for each batchnorm
    #because not inplace anymore
    outputnode = list(gm.graph.nodes)[-1]
    for node in gm.graph.nodes:
        if node.target in ['functional_batch_norm', 'functional_batch_norm_64']:
            with gm.graph.inserting_after(node):
                old_running_mean  = node.args[3]
                old_running_var   = node.args[4]
                new_running_mean  = gm.graph.call_function(operator.getitem, (node, 3))
                new_running_var   = gm.graph.call_function(operator.getitem, (node, 4))
            
            outputlist = list(outputnode.args[0])
            if old_running_mean in outputlist:
                outputlist[outputlist.index(old_running_mean)] = new_running_mean
            if old_running_var in outputlist:
                outputlist[outputlist.index(old_running_var)] = new_running_var
            new_outputnode = gm.graph.output(outputlist)
            outputnode.replace_all_uses_with(new_outputnode)
            gm.graph.erase_node(outputnode)
    gm.graph.lint()
    gm.recompile()


def fixed_squeeze(g, x:torch.Value, dim:torch.Value|None = None):
    if dim is None:
        return g.op('Squeeze', x)
    
    dims_to_squeeze = dim.node().t('value').long()
    if dims_to_squeeze.ndim == 0:
        dims_to_squeeze = dims_to_squeeze[None]
    return g.op("Squeeze", x, g.op("Constant", value_t=dims_to_squeeze))

torch.onnx.register_custom_op_symbolic('::squeeze', fixed_squeeze, opset_version=11)


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
    gm.recompile()

def rename_all_nodes(gm:torch.fx.graph_module.GraphModule, prefix:str):
    for node in gm.graph.nodes:
        node.name = f'{prefix}.{node.name}'



Tensor = torch.Tensor
from enum import Enum

class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def _nll_loss_forward(
    self: Tensor,
    target: Tensor,
    weight: tp.Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> tp.Tuple[Tensor, Tensor]:
    #https://github.com/pytorch/pytorch/blob/7ea184d7e33369610492ff0936369ea00f2b3580/torch/_decomp/decompositions.py#L3269C1-L3319C32
    # self can be [N, C] or [C]
    # target can be [N] or []

    n_dims = self.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    if weight is not None:
        if n_dims > 1:
            shape = [
                1,
            ] * n_dims
            shape[channel_dim] = weight.shape[0]
            w = weight.view(shape)
        else:
            w = weight
        self = self * w
    safe_target = torch.where(target != ignore_index, target, 0)
    safe_target_ = safe_target.unsqueeze(channel_dim)
    # target can be [N, 1] or [1]

    result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)

    #result = torch.where(target != ignore_index, result, 0)
    result = torch.where(
        target != ignore_index, result, torch.as_tensor(0, dtype=torch.float32)
    )

    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = self.new_full((), 0.0)
        return result, total_weight

    if weight is not None:
        w = w.expand(self.shape)
        wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        #wsum = torch.where(target != ignore_index, wsum, 0)
        wsum = torch.where(
            target != ignore_index, wsum, torch.as_tensor(0, dtype=torch.float32)
        )
        total_weight = wsum.sum()
    else:
        total_weight = (target != ignore_index).sum().to(self)

    if reduction == Reduction.SUM.value:
        result = result.sum()
    elif reduction == Reduction.MEAN.value:
        result = result.sum() / total_weight

    return result, total_weight

def _nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: tp.Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    #https://github.com/pytorch/pytorch/blob/7ea184d7e33369610492ff0936369ea00f2b3580/torch/_decomp/decompositions.py#L488C1-L517C36
    channel_dim = 0 if self.dim() < 2 else 1
    if reduction == Reduction.MEAN.value:
        grad_output = grad_output / total_weight

    target = target.unsqueeze(channel_dim)
    safe_target = torch.where(target != ignore_index, target, 0)
    #NOTE: not using zeros_like() because it would be interpreted as a constant 
    #and stored directly in onnx, blowing up the file size
    #grad_input = torch.zeros_like(self)
    grad_input = self * 0.0
    grad_input = torch.scatter(grad_input, channel_dim, safe_target, -1.0)

    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    if weight is not None:
        new_shape = [1 for _ in range(self.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        grad_output = grad_output * weight

    #grad_output = torch.where(target != ignore_index, grad_output, 0)
    grad_output = torch.where(
        target != ignore_index, grad_output, torch.as_tensor(0, dtype=torch.float32)
    )

    return grad_input * grad_output


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

