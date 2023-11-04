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
    ) -> torch.Tensor:
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
    
    
    train_step_tx_0 = make_fx(train_step_0)(sd_grad_vals, sd_nongrad_vals, x)
    #return [train_step_tx_0, sd_grad_vals, sd_nongrad_vals, x,]
    replace_all_conv_backwards(train_step_tx_0)
    replace_all_aten_sgn(train_step_tx_0)
    replace_all_aten_native_batch_norm(train_step_tx_0)
    replace_all_aten_native_batch_norm_backward(train_step_tx_0)

    _params, _mom, _loss, _y, _grads, _nongrads = train_step_tx_0(sd_grad_vals, sd_nongrad_vals, x)
    train_step_tx   = make_fx(train_step)(sd_grad_vals, sd_nongrad_vals, x, _mom)
    replace_all_conv_backwards(train_step_tx)
    replace_all_aten_sgn(train_step_tx)
    replace_all_aten_native_batch_norm(train_step_tx)
    replace_all_aten_native_batch_norm_backward(train_step_tx)

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


def replace_all_conv_backwards(gm:torch.fx.graph_module.GraphModule):
    '''Replace all conv_backward nodes in a traced fx graph with equivalent operations,
       because torch.jit.trace cannot handle it.
    '''
    graph:torch.fx.graph.Graph = gm.graph
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.convolution_backward.default:
            replace_node_with_manual_conv_backwards(graph, node)
    graph.lint()
    gm.recompile()

def replace_node_with_manual_conv_backwards(
    graph:torch.fx.graph.Graph, node:torch.fx.graph.Node
):
    with graph.inserting_before(node):
        gradient                  = node.args[0]
        input: torch.fx.node.Node = node.args[1]  # type: ignore
        weight:torch.fx.node.Node = node.args[2]  # type: ignore
        conv_params               = node.args[4:10]
        
        input_shape     = tuple(input.meta['tensor_meta'].shape) 
        n_groups        = node.args[9]
        input_T         = graph.call_function(torch.ops.aten.transpose, (input,    0,1))
        if n_groups != 1:
            input_T  = graph.call_function(
                torch.reshape,
                (input_T, (input_shape[1]//n_groups, input_shape[0]*n_groups)+input_shape[2:] ) 
            )
        gradient_T      = graph.call_function(torch.ops.aten.transpose, (gradient, 0,1))
        weight_grad_T   = graph.call_function(
            torch.ops.aten.convolution.default, (input_T, gradient_T, None)+conv_params
        )
        weight_grad     = graph.call_function(torch.ops.aten.transpose, (weight_grad_T,    0,1))
        
        gradient_1d     = graph.call_function(
            torch.flatten, (gradient, ), kwargs={'start_dim':2}
        )
        bias_gradient   = graph.call_function(
            torch.ops.aten.sum, (gradient_1d,), kwargs={'dim':[0,2]} 
        )
        
        input_gradient_needed  = node.args[10][0] # type: ignore
        input_gradient  = None
        if input_gradient_needed:
            weight_shape    = weight.meta['tensor_meta'].shape
            if n_groups != 1:
                weight      = graph.call_function(
                torch.reshape, 
                (weight, (weight_shape[0]//n_groups, weight_shape[1]*n_groups) + weight_shape[2:] )
            )
            flip_dims       = list(range(2, len(weight_shape)))
            weight_flip     = graph.call_function(torch.flip, (weight,), kwargs={'dims':flip_dims} )
            weight_flip_T   = graph.call_function(torch.transpose, (weight_flip, 0,1))
            weight_padding  = list(s-1 for s in weight_shape[2:])
            conv2_params    = conv_params[:1] + (weight_padding,) + conv_params[2:]
            input_gradient  = graph.call_function(
                torch.ops.aten.convolution.default, (gradient, weight_flip_T, None)+conv2_params
            )
        
        tuple_op        = graph.call_function(
            tuple, args=([input_gradient, weight_grad, bias_gradient],)
        )
        node.replace_all_uses_with(tuple_op)
    graph.erase_node(node)

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


def replace_node_with_manual_batch_norm_backward(
    graph:torch.fx.graph.Graph, node:torch.fx.graph.Node
):
    #https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    #https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1727
    with graph.inserting_before(node):
        input: torch.fx.node.Node
        grad_out_, input, weight  = node.args[:3]  # type: ignore
        running_mean, running_var = node.args[3:5] #not used here
        save_mean, save_invstd    = node.args[5:7]
        train, eps, grad_in_mask  = node.args[7:]
        assert grad_in_mask == [True,True,True]
        assert train == True

        input_shape = tuple(input.meta['tensor_meta'].shape)
        N,C,H,W = input_shape
        
        d_bias = graph.call_function(
            torch.ops.aten.sum, (grad_out_,), kwargs={'dim':[0,2,3]} 
        )

        save_mean_reshaped   = graph.call_function(torch.reshape, (save_mean, (1,C,1,1)) )
        save_invstd_reshaped = graph.call_function(torch.reshape, (save_invstd, (1,C,1,1)) )
        #TODO: arent xmu and xhat cached from forward pass?
        xmu  = graph.call_function(torch.ops.aten.sub, (input, save_mean_reshaped))
        xhat = graph.call_function(
            torch.ops.aten.mul, (xmu, save_invstd_reshaped)
        )
        grad_mul_xhat = graph.call_function(torch.ops.aten.mul, (grad_out_, xhat))
        sumkwargs = {'dim':[0,2,3]}
        d_weight = graph.call_function(
            torch.ops.aten.sum, (grad_mul_xhat,), kwargs=sumkwargs 
        )

        weight_reshaped = graph.call_function(torch.reshape, (weight, (1,C,1,1)))
        dx_hat          = graph.call_function(torch.ops.aten.mul, (grad_out_, weight_reshaped))
        dx_hat_mul_xmu  = graph.call_function(torch.ops.aten.mul, (dx_hat, xmu))
        keepdims = {'keepdim':True}
        divar  = graph.call_function(
            torch.ops.aten.sum, (dx_hat_mul_xmu,), kwargs=sumkwargs|keepdims
        )
        
        sqrtvar  = graph.call_function(torch.ops.aten.div, (1.0, save_invstd_reshaped))
        var      = graph.call_function(torch.ops.aten.pow, (sqrtvar, 2))
        ninvvar  = graph.call_function(torch.ops.aten.div, (-1.0, var))
        dsqrtvar = graph.call_function(torch.ops.aten.mul, (ninvvar, divar))
        _dvar    = graph.call_function(
            torch.ops.aten.mul, (save_invstd_reshaped, dsqrtvar)
        )
        dvar       = graph.call_function(torch.ops.aten.mul, (0.5, _dvar))
        n_elements = N * H * W
        n_ones     = graph.call_function(torch.ones, (N,C,H,W))
        n_means    = graph.call_function(torch.ops.aten.mul, (1.0/n_elements, n_ones))
        dsq        = graph.call_function(torch.ops.aten.mul, (n_means, dvar))
        xmu2       = graph.call_function(torch.ops.aten.mul, (2.0, xmu))
        dxmu2      = graph.call_function(torch.ops.aten.mul, (xmu2, dsq))

        dxmu1  = graph.call_function(torch.ops.aten.mul, (dx_hat, save_invstd_reshaped))
        dx1    = graph.call_function(torch.ops.aten.add, (dxmu1, dxmu2))
        _dmu   = graph.call_function(torch.ops.aten.sum, (dx1,), kwargs=sumkwargs|keepdims)
        dmu    = graph.call_function(torch.ops.aten.mul, (-1.0, _dmu))
        dx2    = graph.call_function(torch.ops.aten.mul, (n_means, dmu))
        dx     = graph.call_function(torch.ops.aten.add, (dx1, dx2))

        tuple_op = graph.call_function(tuple, args=([dx, d_weight, d_bias],))
        node.replace_all_uses_with(tuple_op)
    graph.erase_node(node)


def find_node_in_graph(graph:torch.fx.graph.Graph, name:str) -> torch.fx.graph.Node|None:
    for node in graph.nodes:
        if node.name == name:
            return node


def replace_node_with_manual_batch_norm_backward(
    graph:torch.fx.graph.Graph, node:torch.fx.graph.Node
):
    #https://github.com/pytorch/pytorch/blob/a44f8894fa6d973693aab44a3dda079a168b05c1/torch/_decomp/decompositions.py#L1727
    empty_args = []
    for arg in node.args[:7]:
        meta = arg.meta['tensor_meta']
        empty_args.append(
            torch.empty(meta.shape, dtype=meta.dtype)
        )
    new_args = empty_args + list(node.args[7:])
    fx = make_fx(torch._decomp.decompositions.native_batch_norm_backward)(*new_args)
    
    node_map = { fxnode:gnode for fxnode,gnode in zip(fx.graph.nodes, node.args)}
    node_map[find_node_in_graph(fx.graph, 'output_mask_1')] = node.args[-1][0]
    node_map[find_node_in_graph(fx.graph, 'output_mask_2')] = node.args[-1][1]
    node_map[find_node_in_graph(fx.graph, 'output_mask_3')] = node.args[-1][2]
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
        meta = arg.meta['tensor_meta']
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


def _early_exit(gm:torch.fx.graph_module.GraphModule, nodename:str):
    '''Create a return statement after a node with specified name and return the node
       (For debugging)'''
    n_leaves = gm._out_spec.num_leaves
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




