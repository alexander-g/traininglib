import typing as tp
import torch
import numpy as np
import io
import operator
import warnings
warnings.simplefilter('ignore', UserWarning)  #pytorch throws too many of those

from torch.fx.experimental.proxy_tensor import make_fx

import sys
#import torch.optim.sgd does not work
sgd_module = sys.modules['torch.optim.sgd']


class ExportedONNX(tp.NamedTuple):
    onnx_bytes_0: bytes
    onnx_bytes:   bytes

def export_model_as_functional_training_onnx(
    m:         torch.nn.Module,
    loss_func: tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x:         torch.Tensor,
) -> ExportedONNX:
    sd                  = {k:v.clone() for k,v in dict(m.state_dict()).items()}
    sd_grad, sd_nongrad = filter_nongrad_values(sd)
    sd_grad_keys        = tuple(sd_grad.keys())
    sd_nongrad_keys     = tuple(sd_nongrad.keys())
    sd_keys             = sd_grad_keys + sd_nongrad_keys
    sd_grad_vals        = tuple(sd_grad.values())
    sd_nongrad_vals     = tuple(sd_nongrad.values())
    
    def forward_step_f(
        sd_grads:    tp.Tuple[torch.nn.Parameter], 
        sd_nongrads: tp.Tuple[torch.nn.Parameter],
        x:           torch.Tensor, 
        t:           torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # torch.jit.trace() complains if a dict is passed to this function
        # but torch.func.functional_call() wants a dict
        sd     = dict(zip(sd_keys, sd_grads + sd_nongrads))
        y      = torch.func.functional_call(m, sd, x, strict=True)
        loss   = loss_func(y, t)
        return loss, y

    #gradient computation function
    grad_f = torch.func.grad_and_value(forward_step_f, argnums=0, has_aux=True)

    def train_step_0(
        sd_grads:    tp.Tuple[torch.nn.Parameter],
        sd_nongrads: tp.Tuple[torch.nn.Parameter],
        x:           torch.Tensor
    ):
        _t = 0 #TODO
        grads, (loss, y) = grad_f(sd_grads, sd_nongrads, x, _t)
        return run_optimizer_init(grads, list(sd_grads)) + (loss, y, grads, sd_nongrads)

    def train_step(
        sd_grads:    tp.Tuple[torch.nn.Parameter],
        sd_nongrads: tp.Tuple[torch.nn.Parameter],
        x:           torch.Tensor,
        mom:         tp.List[torch.Optional[torch.Tensor]],
    ):
        _t = 0 #TODO
        grads, (loss, y) = grad_f(sd_grads, sd_nongrads, x, _t)
        return run_optimizer(grads, list(sd_grads), mom) + (loss, y, grads, sd_nongrads)
    
    decompositions = {
        torch.ops.aten.convolution_backward.default: manual_convolution_backward,
        torch.ops.aten.max_pool2d_with_indices_backward.default: manual_max_pool2d_with_indices_backward,
    }
    train_step_tx_0 = make_fx(train_step_0, decompositions)(sd_grad_vals, sd_nongrad_vals, x)
    
    #return [train_step_tx_0, sd_grad_vals, sd_nongrad_vals, x,]
    replace_all_aten_sgn(train_step_tx_0)
    replace_all_aten_native_batch_norm(train_step_tx_0)
    replace_all_aten_native_batch_norm_backward(train_step_tx_0)
    replace_all_aten_threshold_backward(train_step_tx_0)

    _params, _mom, _loss, _y, _grads, _nongrads = train_step_tx_0(sd_grad_vals, sd_nongrad_vals, x)
    train_step_tx   = make_fx(train_step, decompositions)(sd_grad_vals, sd_nongrad_vals, x, _mom)
    replace_all_aten_sgn(train_step_tx)
    replace_all_aten_native_batch_norm(train_step_tx)
    replace_all_aten_native_batch_norm_backward(train_step_tx)
    replace_all_aten_threshold_backward(train_step_tx)

    inputnames_0  =  [f'p_{k}' for i,k in enumerate(sd_keys)] + ['x']
    inputnames    = ([f'p_{k}' for i,k in enumerate(sd_keys)]
                  +  ['x']
                  +  [f'm_{i}' for i,_ in enumerate(_mom)] )

    outputnames_0 = ([f'p_{k}_' for i,k in enumerate(sd_grad_keys)]
                  +  [f'm_{i}_' for i,_ in enumerate(_mom)]
                  +  ['loss']
                  +  ['y']
                  +  [f'g_{k}_' for i,k in enumerate(sd_grad_keys)]
                  +  [f'p_{k}_' for i,k in enumerate(sd_nongrad_keys)])
    outputnames   = ([f'p_{k}_' for i,k in enumerate(sd_grad_keys)]
                  +  [f'm_{i}_' for i,_ in enumerate(_mom)]
                  +  ['loss']
                  +  ['y']
                  +  [f'g_{k}_' for i,k in enumerate(sd_grad_keys)]
                  +  [f'p_{k}_' for i,k in enumerate(sd_nongrad_keys)])


    #print(train_step_tx_0)
    #print(train_step_tx_0(sd_grad_vals, sd_nongrad_vals, x))
    #torch.jit.trace(train_step_tx_0, (sdvals, x))

    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx_0, 
        (sd_grad_vals, sd_nongrad_vals, x), 
        buf, 
        training     = torch.onnx.TrainingMode.TRAINING,
        input_names  = inputnames_0, 
        output_names = outputnames_0,
        do_constant_folding = False,
    )
    onnx_bytes_0 = buf.getvalue()

    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx, 
        (sd_grad_vals, sd_nongrad_vals, x, _mom), 
        buf, 
        training     = torch.onnx.TrainingMode.TRAINING,
        input_names  = inputnames, 
        output_names = outputnames,
        do_constant_folding = False,
    )
    onnx_bytes = buf.getvalue()

    return ExportedONNX(onnx_bytes_0, onnx_bytes)



def run_optimizer(
    grads:  tp.List[torch.Tensor],
    params: tp.List[torch.Tensor],
    mom:    tp.List[torch.Optional[torch.Tensor]],
):
    sgd_module.sgd(
        params, 
        grads, 
        mom, 
        weight_decay = 1e-4, 
        momentum     = 0.9, 
        lr           = 0.05, 
        dampening    = 0.0, 
        nesterov     = False, 
        maximize     = False,
    )
    return params, mom

def run_optimizer_init(
    grads:  tp.List[torch.Tensor],
    params: tp.List[torch.Tensor],
):
    mom: tp.List[tp.Optional[torch.Tensor]] = [None for _ in params]
    run_optimizer(grads, params, mom)
    return params, mom


StateDict = tp.Dict[str, torch.nn.Parameter]

def state_dict_to_onnx_input(sd:StateDict, onnxnames:tp.List[str]) -> tp.Dict[str, np.ndarray]:
    inputs = { f'p_{k}':v.data.numpy() for k,v in sd.items() }
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
    dilation = tuple(dilation)
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
    #need to truncate, can happen on odd dimensions
    grad_weight = grad_weight[..., :weight.shape[2], :weight.shape[3]]

    grad_out_flat = grad_out.flatten(start_dim=2)
    grad_bias     = grad_out_flat.sum(dim=[0,2])

    grad_input        = None
    grad_input_needed = outmask[0]
    if grad_input_needed:
        if groups != 1:
            weight = weight.reshape(B//groups, C*groups, *weight.shape[2:] )
        flip_dims     = list(range(2, len(weight.shape)))
        weight_flip   = torch.flip(weight, dims=flip_dims)
        weight_flip_T = weight_flip.transpose(0,1)
        padding_gi    = list(s-1 for s in weight.shape[2:])
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
    output      = torch.scatter(z, 2, indices, grad_output)
    return output.reshape(input.shape)


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

def replace_all_aten_threshold_backward(gm:torch.fx.graph_module.GraphModule):
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.threshold_backward.default:
            grad_output, _self, threshold = node.args
            new_args = []
            for argnode in node.args[:2]:
                meta = argnode.meta['tensor_meta']
                new_args += [torch.empty(meta.shape, dtype=meta.dtype)]
            new_args += node.args[2:]
            fx = make_fx(torch._decomp.decompositions.threshold_backward)(*new_args)
            
            node_map = { fxnode:gnode for fxnode,gnode in zip(fx.graph.nodes, node.args)}
            with gm.graph.inserting_before(node):
                newout = gm.graph.graph_copy(fx.graph, node_map, return_output_node=True)
                node.replace_all_uses_with(newout[0])
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
                


def _early_exit(gm:torch.fx.graph_module.GraphModule, nodename:str):
    '''Create a return statement after a node with specified name and return the node
       (For debugging)'''
    n_leaves = gm._out_spec.num_leaves  # type: ignore
    graph:torch.fx.graph.Graph = gm.graph
    for i,node in enumerate(gm.graph.nodes):
        if node.name == nodename:
            with graph.inserting_after(node):
                new_node = graph.output([node]+[node.args]+[None]*(n_leaves-2))
            break
    #remove all following nodes
    for node in reversed(list(gm.graph.nodes)[i+2:]):
        graph.erase_node(node)
    graph.lint()
    gm.recompile()




