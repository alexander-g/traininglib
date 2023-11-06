from traininglib import onnxlib
import onnxruntime as ort
import torch
import numpy as np
import pytest


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
        ), '2dconv & batchnorm'
    ),
    (
        torch.nn.Sequential(
            torch.nn.Conv2d(3,5, kernel_size=3, stride=2),
            torch.nn.MaxPool2d(2, stride=2),
        ), 'stride=2 & maxpool'
    ),
]


@pytest.mark.parametrize("m,desc", testdata)
def test_export(m, desc):
    print(f'==========TEST START: {desc}==========')

    x = torch.randn([2,3,10,10])
    loss_func = lambda y,t: ( (y-1)**2 ).mean()

    exported = onnxlib.export_model_as_functional_training_onnx(m, loss_func, x)

    x = torch.randn([2,3,10,10])

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
        #print('grads torch:', {k:v.numpy().astype('float64') for k,v in grads.items()})

        optim.step()
        optim.zero_grad()

        torch_out = {k:p.data.numpy().copy() for k,p in m.state_dict().items()}
        torch_out['y']    = y.detach().numpy()
        torch_out['loss'] = float(loss)
        torch_outputs.append(torch_out)
        print('=================')
    
    print('*'*50)
    print(out0.keys())
    print(torch_outputs[0].keys())

    for i,[torch_out_i, onnx_out_i] in enumerate(zip(torch_outputs, onnx_outputs)):
        print(f'>>>>>>>>>> train_step: {i} <<<<<<<<<<')

        for [k, p_torch] in torch_out_i.items():
            k_onnx = f'p_{k}_' if k not in ['y', 'loss'] else k
            p_onnx = onnx_out_i[k_onnx]
            print(k)
            print('torch:', np.ravel(p_torch)[-5:], getattr(p_torch, 'shape', None))
            print('onnx: ', np.ravel(p_onnx)[-5:],  getattr(p_onnx,  'shape', None))
            print('diff: ', np.abs(p_torch - p_onnx).max() )
            assert np.allclose(p_torch, p_onnx, atol=1e-06)
            print()
        print()

    print('@'*50)
    #assert 0


def test_conv_backward():
    grad = torch.randn([1,1,4,4])
    inp  = torch.randn([1,2,10,10])
    wgt  = torch.randn([1,2,3,3])

    args = (grad, inp, wgt, [4], [2,2], [0,0], [1,1], False, [0,0], 1, [True,True,True])
    my_out    = onnxlib.manual_convolution_backward(*args)
    torch_out = torch.ops.aten.convolution_backward.default(*args)
    for t_out, m_out in zip(torch_out, my_out):
        print('torch: ', t_out.numpy().round(2))
        print('manual:', m_out.numpy().round(2))
        if t_out is None:
            assert m_out is None
            continue
        assert torch.allclose(t_out, m_out)
        print()

    #assert 0