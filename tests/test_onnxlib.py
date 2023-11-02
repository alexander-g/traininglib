from traininglib import onnxlib
import onnxruntime as ort
import torch
import numpy as np



def test_export():
    m = torch.nn.Sequential(
        torch.nn.Conv2d(3,5, kernel_size=3),
        torch.nn.Conv2d(5,1, kernel_size=1),
    )
    x = torch.randn([2,3,10,10])
    loss_func = lambda y,t: ( (y-1)**2 ).mean()

    exported = onnxlib.export_model_as_functional_training_onnx(m, loss_func, x)

    x = torch.randn([2,3,10,10])
    
    sess0    = ort.InferenceSession(exported.onnx_bytes_0)
    outputs0 = [o.name for o in sess0.get_outputs()]
    inputs0  = onnxlib.state_dict_to_onnx_input(m.state_dict())
    inputs0.update({'x': x.numpy()})
    out0     = sess0.run(outputs0, inputs0)
    out0     = dict(zip(outputs0, out0))
    del sess0

    sess1       = ort.InferenceSession(exported.onnx_bytes)
    outputs1    = [o.name for o in sess1.get_outputs()]
    inputnames1 = [i.name for i in sess1.get_inputs() if not i.name in ['x']]
    inputs1      = {k:out0[f'{k}_'] for k in inputnames1}
    inputs1.update(x = x.numpy())
    out1        = sess1.run(outputs1, inputs1)
    out1        = dict(zip(outputs1, out1))
    

    torch_outs = []
    optim = torch.optim.SGD(
        m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4, dampening=0.0, nesterov=False
    )
    for i in range(2):
        y    = m.train()(x)
        loss = loss_func(y, None)
        print(loss.item())
        loss.backward()

        grads = {k:p.grad for k,p in zip(m.state_dict().keys(), m.parameters()) }
        print('grads torch:', {k:v.numpy().astype('float64') for k,v in grads.items()})

        optim.step()
        optim.zero_grad()

        #torch_outs.append( [p.data.numpy()  for p in m.parameters()] )
        torch_outs.append([p.data.numpy().copy() for p in m.state_dict().values()])
    
    print('*'*50)

    print()
    print( float(out0['loss']) )
    print( 'grads onnx:', list(out0.items())[-2:] )
    print( float(out1['loss']) )
    print( 'grads onnx:', list(out1.items())[-2:] )

    print()
    print('*'*50)
    for a,b in zip(torch_outs[0], out0.values()):
        print('torch:', a[0])
        print('onnx:',  b[0])
        assert np.allclose(a,b)
        print()
    
    for a,b in zip(torch_outs[1], out1.values()):
        print('torch:', a[0])
        print('onnx:',  b[0])
        assert np.allclose(a,b)
        print()



    #assert 0

