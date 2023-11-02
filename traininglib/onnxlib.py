import typing as tp
import torch
import numpy as np
import io
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
    sd     = dict(m.state_dict())
    sdkeys = tuple(sd.keys())
    sdvals = tuple(sd.values())
    
    def forward_step_f(
        sd_tuple: tp.Tuple[torch.nn.Parameter], 
        x:        torch.Tensor, 
        t:        torch.Tensor,
    ) -> torch.Tensor:
        # torch.jit.trace() complains if a dict is passed to this function
        # but torch.func.functional_call() wants a dict
        sd     = dict(zip(sdkeys, sd_tuple))
        y      = torch.func.functional_call(m, sd, x, strict=True)
        loss   = loss_func(y, t)
        return loss

    #gradient computation function
    grad_f = torch.func.grad_and_value(forward_step_f, argnums=0)

    def train_step_0(
        sd_tuple:tp.Tuple[torch.nn.Parameter],
        x:       torch.Tensor
    ):
        _t = 0 #TODO
        grads, loss = grad_f(sd_tuple, x, _t)
        return run_optimizer_init(grads, list(sd_tuple)) + (loss, grads)

    def train_step(
        sd_tuple:tp.Tuple[torch.nn.Parameter],
        x:       torch.Tensor,
        mom:     tp.List[torch.Optional[torch.Tensor]],
    ):
        _t = 0 #TODO
        grads, loss = grad_f(sd_tuple, x, _t)
        return run_optimizer(grads, list(sd_tuple), mom) + (loss, grads)
    
    
    train_step_tx_0 = make_fx(train_step_0)(sdvals, x)
    _params, _mom, _loss, _grads = train_step_tx_0(sdvals, x)
    train_step_tx   = make_fx(train_step)(sdvals, x, _mom)

    replace_all_conv_backwards(train_step_tx_0)
    replace_all_aten_sgn(train_step_tx_0)
    replace_all_conv_backwards(train_step_tx)
    replace_all_aten_sgn(train_step_tx)

    inputnames_0  =  [f'p_{k}' for i,k in enumerate(sdkeys)] + ['x']
    inputnames    = ([f'p_{k}' for i,k in enumerate(sdkeys)]
                  +  ['x']
                  +  [f'm_{i}' for i,_ in enumerate(_mom)] )

    outputnames_0 = ([f'p_{k}_' for i,k in enumerate(sdkeys)]
                  +  [f'm_{i}_' for i,_ in enumerate(_mom)]
                  +  ['loss']
                  +  [f'g_{k}_' for i,k in enumerate(sdkeys)])
    outputnames   = ([f'p_{k}_' for i,k in enumerate(sdkeys)]
                  +  [f'm_{i}_' for i,_ in enumerate(_mom)]
                  +  ['loss']
                  +  [f'g_{k}_' for i,k in enumerate(sdkeys)])

    print(train_step_tx_0)
    print(train_step_tx_0(sdvals, x))
    #torch.jit.trace(train_step_tx_0, (sdvals, x))

    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx_0, (sdvals, x), buf, input_names=inputnames_0, output_names=outputnames_0,
    )
    onnx_bytes_0 = buf.getvalue()

    buf = io.BytesIO()
    torch.onnx.export(
        train_step_tx, (sdvals,  x, _mom), buf, input_names=inputnames, output_names=outputnames,
    )
    onnx_bytes = buf.getvalue()

    return ExportedONNX(onnx_bytes_0, onnx_bytes)


def state_dict_to_onnx_input(sd:tp.Dict[str, torch.nn.Parameter]) -> tp.Dict[str, np.ndarray]:
    return {
        f'p_{k}':v.data.numpy() for k,v in sd.items()
    }



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
