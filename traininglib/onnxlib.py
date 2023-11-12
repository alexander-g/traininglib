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
        x:  torch.Tensor
    ) -> tp.Dict[str, torch.Tensor|TensorDict|StateDict]:
        _t = 0 #TODO
        sd_grads, sd_nongrads = filter_nongrad_values(sd)
        gradients, (loss, y)  = grad_f(sd_grads, sd_nongrads, x, _t)
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
        buffers: TensorDict,
    ):
        _t = 0 #TODO
        sd_grads, sd_nongrads = filter_nongrad_values(sd)
        gradients, (loss, y)  = grad_f(sd_grads, sd_nongrads, x, _t)
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
        torch.ops.aten.convolution_backward.default: manual_convolution_backward,
        torch.ops.aten.max_pool2d_with_indices_backward.default: manual_max_pool2d_with_indices_backward,
        torch.ops.aten.threshold_backward.default: threshold_backward,
        torch.ops.aten.native_batch_norm_backward.default: torch._decomp.decompositions.native_batch_norm_backward,
        torch.ops.aten.var_mean.correction:var_mean_64,

    }
    train_step_tx_0 = make_fx(train_step_0, tracing_mode='fake')(sd, x)
    rename_all_nodes(train_step_tx_0, prefix='main')
    replace_all_aten_sgn(train_step_tx_0)
    replace_all_aten_native_batch_norm(train_step_tx_0)
    replace_all_inplace_add(train_step_tx_0)
    replace_all_inplace_mul(train_step_tx_0)
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(train_step_tx_0, torch_op, manual_op)
    if _debug:
        train_step_tx_0 = _return_all_nodes(train_step_tx_0)

        #unmodified, i.e. no decompositions
        train_step_tx_0_unmodified = make_fx(train_step_0, tracing_mode='fake')(sd, x)
        rename_all_nodes(train_step_tx_0_unmodified, prefix='main')
        #modified after all, removing inplace ops
        replace_all_aten_native_batch_norm(train_step_tx_0_unmodified)
        replace_all_inplace_add(train_step_tx_0_unmodified)
        replace_all_inplace_mul(train_step_tx_0_unmodified)
        train_step_tx_0_unmodified = _return_all_nodes(train_step_tx_0_unmodified)


    _init_out = train_step_tx_0(sd, x)
    train_step_tx   = make_fx(train_step, tracing_mode='fake')(
        sd, x, _init_out['buffers']
    )
    rename_all_nodes(train_step_tx, prefix='main')
    replace_all_aten_sgn(train_step_tx)
    replace_all_aten_native_batch_norm(train_step_tx)
    replace_all_inplace_add(train_step_tx)
    replace_all_inplace_mul(train_step_tx)
    for torch_op, manual_op in decompositions.items():
        replace_op_type_with_custom_function(train_step_tx, torch_op, manual_op)
    if _debug:
        train_step_tx = _return_all_nodes(train_step_tx)

        #unmodified, i.e. no decompositions
        train_step_tx_unmodified = make_fx(train_step, tracing_mode='fake')(
            sd, x, _init_out['buffers']
        )
        rename_all_nodes(train_step_tx_unmodified, prefix='main')
        #modified after all, removing inplace ops
        replace_all_aten_native_batch_norm(train_step_tx_unmodified)
        replace_all_inplace_add(train_step_tx_unmodified)
        replace_all_inplace_mul(train_step_tx_unmodified)
        train_step_tx_unmodified = _return_all_nodes(train_step_tx_unmodified)


    inputnames_0  = list(sd.keys()) + ['x']
    inputnames    = inputnames_0 + [f'{k}.buffer' for k in _init_out['buffers'].keys()]
    outputnames_0 = [f'{k}.output' for k in _init_out['state_dict'].keys()]         \
                  + [f'{k}.buffer.output' for k in _init_out['buffers'].keys()]     \
                  + [f'{k}.gradient.output' for k in _init_out['gradients'].keys()] \
                  + ['loss', 'y']
    outputnames   = list(outputnames_0)
    if _debug:
        outputnames_0 += [f'{k}.debug' for k in _init_out['debug'].keys()]
        
        _2nd_out = train_step_tx_unmodified(sd, x, _init_out['buffers'])
        outputnames   += [f'{k}.debug' for k in _2nd_out['debug'].keys()]


    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx_0, 
        {'sd':sd, 'x':x},
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
        {'sd':sd, 'x':x, 'buffers':_init_out['buffers']},
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
    b = torch.zeros([B,C,H,W*2])
    b = torch.scatter(b, 3, indices, x)

    indices = torch.arange(H)*2
    indices = indices.reshape(1,1,-1,1)
    indices = indices.expand(B,C,-1,W*2)
    c = torch.zeros([B,C,H*2,W*2])
    c = torch.scatter(c, 2, indices, b)

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
    z = torch.zeros_like(input.reshape(intermediate_shape))
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


def replace_all_aten_native_batch_norm_backward(gm:torch.fx.graph_module.GraphModule):
    '''Replace all conv_backward nodes in a traced fx graph with equivalent operations,
       because torch.onnx.export cannot handle it.
    '''
    graph:torch.fx.graph.Graph = gm.graph
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.native_batch_norm_backward.default:
            replace_node_with_manual_batch_norm_backward(graph, node)
    graph.lint()
    gm.recompile()


def find_node_in_graph(graph:torch.fx.graph.Graph, name:str) -> torch.fx.graph.Node|None:
    for node in graph.nodes:
        if node.name == name:
            return node
    return None


def replace_node_with_manual_batch_norm_backward(
    graph:torch.fx.graph.Graph, node:torch.fx.graph.Node
):
    #https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1727
    empty_args = []
    for arg in node.args[:7]:
        meta = arg.meta['tensor_meta']  # type: ignore
        empty_args.append(
            torch.empty(meta.shape, dtype=meta.dtype)
        )
    new_args = empty_args + list(node.args[7:])
    fx = make_fx(torch._decomp.decompositions.native_batch_norm_backward)(*new_args)
    
    node_map = { fxnode:gnode for fxnode,gnode in zip(fx.graph.nodes, node.args)}
    node_map[find_node_in_graph(fx.graph, 'output_mask_1')] = node.args[-1][0] # type: ignore
    node_map[find_node_in_graph(fx.graph, 'output_mask_2')] = node.args[-1][1] # type: ignore
    node_map[find_node_in_graph(fx.graph, 'output_mask_3')] = node.args[-1][2] # type: ignore
    with graph.inserting_before(node):
        out = graph.graph_copy(fx.graph, node_map, return_output_node=True)
        fx_outputs  = list(fx.graph.nodes)[-1].args[0]
        new_outputs = tuple(node_map[n] for n in fx_outputs)
        tuple_op = graph.call_function(tuple, args=(new_outputs,))
        node.replace_all_uses_with(tuple_op)
    graph.erase_node(node)
    
    graph.lint()
    graph.owning_module.recompile()


def replace_all_aten_native_batch_norm(gm:torch.fx.graph_module.GraphModule):
    '''Replace all aten_native_batch_norm nodes in a traced fx graph with aten_batch_norm'''
    graph:torch.fx.graph.Graph = gm.graph
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.native_batch_norm.default:
            replace_node_with_aten_native_batch_norm(gm.graph, node)
    graph.lint()
    gm.recompile()

    replace_all_squeeze_dims(gm)


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
    fx = make_fx(torch._decomp.decompositions.native_batch_norm_helper)(*new_args)
    fx.graph.erase_node(
        find_node_in_graph(fx.graph, 'functional_1')
    )

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
                node.replace_all_uses_with(new_squeeze)
            graph.erase_node(node)
    graph.lint()
    gm.recompile()

def replace_all_inplace_add(gm:torch.fx.graph_module.GraphModule):
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.add_.Tensor:
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(torch.ops.aten.add, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()

def replace_all_inplace_mul(gm:torch.fx.graph_module.GraphModule):
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.mul_.Tensor:
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(torch.ops.aten.mul, node.args, node.kwargs)
                node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()


def rename_all_nodes(gm:torch.fx.graph_module.GraphModule, prefix:str):
    for node in gm.graph.nodes:
        node.name = f'{prefix}.{node.name}'

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

