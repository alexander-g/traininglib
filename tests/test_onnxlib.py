import typing as tp
import functools
from traininglib import onnxlib
import onnxruntime as ort
import torch, torchvision
import numpy as np
import pytest


class MiniResNet(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(3,8, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(),
        )

class MiniMobileNet(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torchvision.ops.misc.Conv2dNormActivation(
                3,
                16,
                kernel_size=3,
                stride=2,
                norm_layer=functools.partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=torch.nn.Hardswish,
                #activation_layer=lambda *a,**kw: torch.nn.Hardswish(inplace=False),
            ),
            torchvision.models.mobilenetv3.InvertedResidual(
                torchvision.models.mobilenetv3.InvertedResidualConfig(
                    16, 3, 16, 16, True, "RE", 1, 1, 1
                ),
                functools.partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.01),
            ),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(),
        )
    

class TestItem(tp.NamedTuple):
    module:    torch.nn.Module
    loss_func: tp.Callable
    x:         torch.Tensor
    t:         torch.Tensor
    atol:      float         = 1e-6

testdata: tp.List[tp.Tuple[TestItem, str]] = [
    (
        TestItem(
            module = torch.nn.Sequential(
                torch.nn.Conv2d(3,5, kernel_size=3),
                torch.nn.Conv2d(5,1, kernel_size=1),
            ),
            loss_func = torch.nn.functional.l1_loss,
            x         = torch.rand([2,3,10,10]),
            t         = torch.rand([2,1,8,8])
        ) , '2x-2dconv'
    ),
    (
        TestItem(
            module = torch.nn.Sequential(
                torch.nn.Conv2d(3,5, kernel_size=3),
                torch.nn.BatchNorm2d(5),
                torch.nn.ReLU(),
                torch.nn.Conv2d(5,1, kernel_size=1),
            ),
            #loss_func = torch.nn.functional.mse_loss,      #TODO
            loss_func = torch.nn.functional.l1_loss,
            x         = torch.rand([2,3,10,10]),
            t         = torch.rand([2,1,8,8]),
        ), '2dconv-batchnorm'
    ),
    (
        TestItem(
            module = torch.nn.Sequential(
                torch.nn.Conv2d(3,5, kernel_size=3, stride=2),
                torch.nn.MaxPool2d(2, stride=2),
            ),
            loss_func = torch.nn.functional.l1_loss,
            x         = torch.rand([2,3,10,10]),
            t         = torch.rand([2,1,2,2])
        ), 'stride2-maxpool'
    ),
    (
        TestItem(
            module    = MiniResNet(),
            loss_func = torch.nn.functional.cross_entropy,
            x         = torch.rand([2,3,10,10]),
            t         = torch.randint(0,8, [2]),
        ), 'miniresnet'
    ),
    (
        TestItem(
            module    = MiniMobileNet(),
            loss_func = torch.nn.functional.cross_entropy,
            x         = torch.rand([2,3,10,10]),
            t         = torch.randint(0,5, [2]),
        ), 'mini-mobilenet-v3'
    ),
    # (
    #     TestItem(
    #         module    = torchvision.models.resnet._resnet(
    #            torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], None, False
    #         ),
    #         loss_func = torch.nn.functional.cross_entropy,
    #         x         = torch.rand([2,3,10,10]),
    #         t         = torch.randint(0,5, [2]),
    #         atol      = 1e-2,
    #     ), 'resnet10?'
    # ),
    # (
    #     TestItem(
    #         #module    = torchvision.models.mobilenet_v3_large(weights=None, progress=False),
    #         module    = torchvision.models.mobilenet_v3_small(weights=None, progress=False),
    #         loss_func = torch.nn.functional.cross_entropy,
    #         x         = torch.rand([2,3,10,10]),
    #         t         = torch.randint(0,5, [2]),

    #     ), 'mobilenet-v3-full'
    # ),
    #(torchvision.models.resnet18(weights='DEFAULT', progress=None), 'resnet18'),
]


@pytest.mark.parametrize("testitem,desc", testdata)
def test_export(testitem:TestItem, desc:str):
    print(f'==========TEST START: {desc}==========')

    m = testitem.module
    x = testitem.x
    t = testitem.t
    loss_func = testitem.loss_func

    x0 = torch.randn(x.shape)
    exported = onnxlib.export_model_as_functional_training_onnx(m, loss_func, x0, t)


    onnx_outputs = []

    sess0        = ort.InferenceSession(exported.onnx_bytes_0)
    outputnames0 = [o.name for o in sess0.get_outputs()]
    inputsnames0 = [o.name for o in sess0.get_inputs()]
    inputs0 = exported.inputfeed
    inputs0.update({'x': x.numpy(), 't':t.numpy()})
    out0     = sess0.run(outputnames0, inputs0)
    out0     = dict(zip(outputnames0, out0))
    onnx_outputs.append(out0)
    del sess0

    sess1        = ort.InferenceSession(exported.onnx_bytes)
    outputnames1 = [o.name for o in sess1.get_outputs()]
    inputnames1  = [i.name for i in sess1.get_inputs()]
    for i in range(3):
        prev_out     = onnx_outputs[-1]
        inputs1      = {k:prev_out[f'{k}.output'] for k in inputnames1 if k not in ['x', 't']}
        inputs1.update(x = x.numpy(), t = t.numpy())
        out1        = sess1.run(outputnames1, inputs1)
        out1        = dict(zip(outputnames1, out1))
        onnx_outputs.append(out1)
    
    print('onnx y0:   ', out0['y'].sum(-1).sum(-1).ravel())
    print('onnx loss0:', out0['loss'])
    print('onnx y1:   ', out1['y'].sum(-1).sum(-1).ravel())
    print('onnx loss1:', out1['loss'])
    print('------------')


    torch_outputs = []
    optim = torch.optim.SGD(
        m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, dampening=0.0, nesterov=False
    )
    for i in range(len(onnx_outputs)):
        y    = m.train()(x)
        print(f'torch y{i}:', y.detach().numpy().sum(-1).sum(-1).ravel())
        loss = loss_func(y, t)
        print('torch loss:', loss.item())
        loss.backward()

        #grads = {k:p.grad for k,p in zip(m.state_dict().keys(), m.parameters()) }
        grads = {k:p.grad for k,p in m.named_parameters() }
        grads = {k:v for k,v in grads.items() if 'running' not in k}
        grads = {k:v for k,v in grads.items() if v is not None}

        optim.step()
        optim.zero_grad()

        torch_out = {}
        #forward pass output
        torch_out['y']    = y.detach().numpy()
        #loss
        torch_out['loss'] = float(loss)
        #gradients
        torch_out.update(
            {f'{k}.gradient.output':v.numpy().astype('float64') for k,v in reversed(grads.items())}
        )
        #parameters
        torch_out.update(
            {f'{k}.output':p.data.numpy().copy() for k,p in reversed(m.state_dict().items())}
        )
        torch_outputs.append(torch_out)
        print('=================')
    
    print('*'*50)
    print(out0.keys())
    print(torch_outputs[0].keys())

    for i,[torch_out_i, onnx_out_i] in enumerate(zip(torch_outputs, onnx_outputs)):
        print(f'>>>>>>>>>> train_step: {i} <<<<<<<<<<')

        for [k, p_torch] in torch_out_i.items():
            k_onnx = k
            p_onnx = onnx_out_i[k_onnx]
            print(k)
            print('torch:', np.ravel(p_torch)[-16:], getattr(p_torch, 'shape', None))
            print('onnx: ', np.ravel(p_onnx)[-16:],  getattr(p_onnx,  'shape', None))
            print('diff: ', np.abs(p_torch - p_onnx).max() )
            assert np.allclose(p_torch, p_onnx, atol=testitem.atol)
            print()
        print()

    print('@'*50)
    #assert 0


conv_testdata = [
    ([
        torch.randn([1,1,4,4]), torch.randn([1,2,10,10]), torch.randn([1,2,3,3]),
        [4], [2,2], [0,0], [1,1], False, [0,0], 1, [True,True,True]
    ], 'stride=2'),
    ([
        torch.randn([2,10,1,1]), torch.randn([2,5,2,2]), torch.randn([10,5,3,3]),
        [4], [2,2], [1,1], [1,1], False, [0,0], 1, [True,True,True]
    ], 'stride=2,padding=1'),
    ([
        torch.randn([2,10,2,2]), torch.randn([2,5,3,3]), torch.randn([10,5,1,1]),
        [4], [2,2], [0,0], [1,1], False, [0,0], 1, [True,True,True]
    ], 'need-to-truncate-shape'),
    ([
        torch.rand([2,8,5,5]), torch.rand([2,3,10,10]), torch.rand([8,3,7,7]),
        [8], [2,2], [3,3], [1,1], False, [0,0], 1, [False,True,True]
    ], 'resnet-first-layer'),
    ([
        torch.rand([2,96,1,1]), torch.rand([2,96,1,1]), torch.rand([96,1,5,5]),
        [0], [1,1], [2,2], [1,1], False, [0,0], 96, [True,True,True]
    ], 'mobilenet-with-groups')
]

@pytest.mark.parametrize("args, desc", conv_testdata)
def test_conv_backward(args, desc):
    print(f'==========TEST START: {desc}==========')
    my_out    = onnxlib.manual_convolution_backward(*args)
    torch_out = torch.ops.aten.convolution_backward.default(*args)
    for t_out, m_out in zip(torch_out, my_out):
        if t_out is None:
            assert m_out is None
            continue
        m_out = tp.cast(torch.Tensor, m_out)
        print('torch: ', t_out.numpy().round(2), t_out.shape)
        print('manual:', m_out.numpy().round(2), m_out.shape)
        print('diff: ', np.abs(t_out.numpy() - m_out.numpy()).max() )
        assert torch.allclose(t_out, m_out, atol=1e-7)
        print()

    #assert 0



maxpool_testdata = [
    ([
        torch.randn([1,5,2,1]), torch.randn(1,5,4,3), [2,2], [2,2], [0,0], [1,1], False,
        torch.randint(0,2, [1,5,2,1])
    ], 'maxpool2d.basic'),
    ([
        torch.randn([2,8,3,3]), torch.randn([2,8,5,5]), [3,3], [2,2], [1,1], [1,1], False,
        torch.randint(0,24, [2,8,3,3]),
    ], 'maxpool2d.kernel3x3')
]

@pytest.mark.parametrize("args, desc", maxpool_testdata)
def test_maxpool_backward(args, desc):
    print(f'==========TEST START: {desc}==========')
    print(args[0].numpy().round(2))
    print(args[-1][0])
    torch_out = torch.ops.aten.max_pool2d_with_indices_backward.default(*args)
    my_out    = onnxlib.manual_max_pool2d_with_indices_backward(*args)
    print('torch: ', torch_out.numpy().round(2), torch_out.shape)
    print('manual:', my_out.numpy().round(2), my_out.shape)
    assert torch.allclose(torch_out, my_out)
    print()