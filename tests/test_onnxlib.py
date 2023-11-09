from traininglib import onnxlib
import onnxruntime as ort
import torch, torchvision
import numpy as np
import pytest


class MiniResNet(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(3,8, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            #torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            torch.nn.AdaptiveAvgPool2d((1,1)),
        )

testdata = [
    (
        torch.nn.Sequential(
            torch.nn.Conv2d(3,5, kernel_size=3),
            torch.nn.Conv2d(5,1, kernel_size=1),
        ) , '2x 2dconv'
    ),
    (
        torch.nn.Sequential(
            torch.nn.Conv2d(3,5, kernel_size=3),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(5,1, kernel_size=1),
        ), '2dconv-batchnorm'
    ),
    (
        torch.nn.Sequential(
            torch.nn.Conv2d(3,5, kernel_size=3, stride=2),
            torch.nn.MaxPool2d(2, stride=2),
        ), 'stride=2-maxpool'
    ),
    (MiniResNet(), 'miniresnet'),
    #(torchvision.models.resnet18(weights=None, progress=None), 'resnet18'),
]


@pytest.mark.parametrize("m,desc", testdata)
def test_export(m, desc):
    print(f'==========TEST START: {desc}==========')

    x = torch.randn([2,3,10,10])
    loss_func = lambda y,t: ( (y-1)**2 ).mean()

    exported = onnxlib.export_model_as_functional_training_onnx(m, loss_func, x)

    x = torch.rand(x.shape) * 0

    onnx_outputs = []

    sess0    = ort.InferenceSession(exported.onnx_bytes_0)
    outputs0 = [o.name for o in sess0.get_outputs()]
    inputsnames0  = [o.name for o in sess0.get_inputs()]
    inputs0  = onnxlib.state_dict_to_onnx_input(m.state_dict(), inputsnames0)
    inputs0.update({'x': x.numpy()})
    out0     = sess0.run(outputs0, inputs0)
    out0     = dict(zip(outputs0, out0))
    onnx_outputs.append(out0)
    del sess0

    sess1       = ort.InferenceSession(exported.onnx_bytes)
    outputs1    = [o.name for o in sess1.get_outputs()]
    inputnames1 = [i.name for i in sess1.get_inputs() if not i.name in ['x']]
    for i in range(3):
        prev_out     = onnx_outputs[-1]
        inputs1      = {k:prev_out[f'{k}_'] for k in inputnames1}
        inputs1.update(x = x.numpy())
        out1        = sess1.run(outputs1, inputs1)
        out1        = dict(zip(outputs1, out1))
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
        loss = loss_func(y, None)
        print('torch loss:', loss.item())
        loss.backward()

        grads = {k:p.grad for k,p in zip(m.state_dict().keys(), m.parameters()) }
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
            {f'g_{k}_':v.numpy().astype('float64') for k,v in reversed(grads.items())}
        )
        #parameters
        torch_out.update(
            {f'p_{k}_':p.data.numpy().copy() for k,p in reversed(m.state_dict().items())}
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
            assert np.allclose(p_torch, p_onnx, atol=1e-06)
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
    ], 'resnet-first-layer')
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
        print('torch: ', t_out.numpy().round(2), t_out.shape)
        print('manual:', m_out.numpy().round(2), m_out.shape)
        print('diff: ', np.abs(t_out.numpy() - m_out.numpy()).max() )
        assert torch.allclose(t_out, m_out)
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