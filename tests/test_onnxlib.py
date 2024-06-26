import typing as tp
import functools
import json
import os
import tempfile
import zipfile

from traininglib import onnxlib, unet
import onnxruntime as ort
import torch, torchvision
import numpy as np
import pytest


#NOTE: this has to come first because of issues with custom onnx op registration
#TODO: fix those issues
def test_export_faster_rcnn():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, backbone_weights=None, progress=None
    )
    x = torch.randn(1, 3, 224, 224)
    exported = onnxlib.export_model_inference(
        model, x, outputnames=['boxes', 'labels', 'scores']
    )

    tempdir  = tempfile.TemporaryDirectory()
    temppath = os.path.join(tempdir.name, 'model.pt.zip')
    exported.save_as_zipfile(temppath)

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(exported.onnx_bytes, session_options)
    outputs = session.run(['boxes', 'scores'], exported.inputfeed|{'x':x.numpy()})
    assert len(outputs) == 2



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

default_sgd = lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-4)

class TestItem(tp.NamedTuple):
    module:    torch.nn.Module
    x:         torch.Tensor
    loss_func: tp.Callable   = lambda: None
    t:         torch.Tensor  = torch.as_tensor(0)
    optim:     tp.Callable   = default_sgd
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
                torch.nn.Conv2d(5,1, kernel_size=1),
            ),
            loss_func = torch.nn.functional.l1_loss,
            x         = torch.rand([2,3,10,10]),
            t         = torch.rand([2,1,8,8]),
            optim     = lambda p: torch.optim.AdamW(p, lr=2e-4),
        ) , 'adamw'
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
]


@pytest.mark.parametrize("testitem,desc", testdata)
def test_training(testitem:TestItem, desc:str):
    print(f'==========TRAINING TEST: {desc}==========')

    m = testitem.module
    x = testitem.x
    t = testitem.t
    loss_func = testitem.loss_func
    optimizer = testitem.optim(m.parameters())

    x0 = torch.randn(x.shape)
    exported = onnxlib.export_model_as_functional_training_onnx(
        m, loss_func, x0, t, optimizer
    )

    onnx_outputs:tp.List[tp.Any] = []

    sess1        = ort.InferenceSession(exported.onnx_bytes)
    outputnames1 = [o.name for o in sess1.get_outputs()]
    inputnames1  = [i.name for i in sess1.get_inputs()]
    for i in range(4):
        if len(onnx_outputs):
            prev_out = onnx_outputs[-1]
            inputs1  = {
                k:prev_out[f'{k}.output'] for k in inputnames1 if k not in ['x', 't']
            }
        else:
            inputs1  = exported.inputfeed
        inputs1.update(x = x.numpy(), t = t.numpy())
        out1        = sess1.run(outputnames1, inputs1)
        out1        = dict(zip(outputnames1, out1))
        onnx_outputs.append(out1)
    
    for i,out_i in enumerate(onnx_outputs):
        print(f'onnx y{i}:   ', out_i['y'].sum(-1).sum(-1).ravel())
        print(f'onnx loss{i}:', out_i['loss'])
    print('------------')


    torch_outputs = []
    optim = testitem.optim(m.parameters())
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
            {f'{k}.gradient.output':v.numpy().astype('float64')  # type: ignore
                for k,v in reversed(grads.items())}
        )
        #parameters
        torch_out.update(
            {f'{k}.output':p.data.numpy().copy() for k,p in reversed(m.state_dict().items())}
        )
        torch_outputs.append(torch_out)
    
    print('*'*50)
    #print(out0.keys())
    #print(torch_outputs[0].keys())

    for i,[torch_out_i, onnx_out_i] in enumerate(zip(torch_outputs, onnx_outputs)):
        print(f'>>>>>>>>>> train_step: {i} <<<<<<<<<<')

        for [k, p_torch] in torch_out_i.items():
            k_onnx = k
            p_onnx = onnx_out_i[k_onnx]
            print(k)
            print('torch:', np.ravel(p_torch)[-5:], getattr(p_torch, 'shape', None))
            print('onnx: ', np.ravel(p_onnx)[-5:],  getattr(p_onnx,  'shape', None))
            print('diff: ', np.abs(p_torch - p_onnx).max() )
            assert np.allclose(p_torch, p_onnx, atol=testitem.atol)
            print()
        print()

    print('@'*50)
    #assert 0



inference_testdata = testdata + [
(
    TestItem(
        module    = unet.UNet(backbone='mobilenet3l', backbone_weights=None),
        x         = torch.rand([1,3,64,64]),
    ), 'unet'
)
]

@pytest.mark.parametrize("testitem,desc", inference_testdata)
def test_inference(testitem:TestItem, desc:str):
    print(f'==========INFERENCE TEST: {desc}==========')
    m = testitem.module
    x = testitem.x

    exported = onnxlib.export_model_as_functional_inference_onnx(m, x)
    session  = ort.InferenceSession(exported.onnx_bytes)

    outputnames = [o.name for o in session.get_outputs()]
    inputfeed   = exported.inputfeed | {'x':x.numpy()}
    onnx_output = session.run(outputnames, inputfeed)

    torch_output = m.eval().requires_grad_(False)(x).numpy()

    diff = np.abs(onnx_output - torch_output).max()
    print('diff:', diff)
    assert np.allclose(onnx_output, torch_output, atol=1e-7)

    tempdir = tempfile.TemporaryDirectory()
    exported.save_as_zipfile(os.path.join(tempdir.name, 'exported.pt.zip'))

def test_inference_dynamic_shape():
    m = torch.nn.Conv2d(3,1, kernel_size=3)
    x = torch.ones([1,3,128,128])

    exported = onnxlib.export_model_inference(
        m, x, ['x'], ['y'], dynamic_axes={'x':[2]}
    )
    tempdir = tempfile.TemporaryDirectory()
    tempf   = os.path.join(tempdir.name, 'dynamic.pt.zip')
    exported.save_as_zipfile(tempf)

    with zipfile.ZipFile(tempf) as zipf:
        onnx_file = [f for f in zipf.namelist() if f.endswith('inference.onnx')][0]
        session = ort.InferenceSession(zipf.read(onnx_file))
        assert session.get_inputs()[0].shape == [1,3,'x_dynamic_axes_1',128]

        schema_file = [f for f in zipf.namelist() if f.endswith('inference.schema.json')][0]
        jsondata = json.loads(zipf.read(schema_file))
        assert jsondata['x']['shape'] == [1,3,None,128]





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
        assert torch.allclose(t_out, m_out, atol=5e-6)
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

