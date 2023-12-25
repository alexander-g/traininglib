import typing as tp
import torch
import numpy as np

from traininglib import torchscriptlib

class ConvAndLoss(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.conv = torch.nn.Conv2d(3,1, kernel_size=3, padding=1)

        def forward(self, x:tp.List[torch.Tensor], t:torch.Tensor):
            output = self.conv(x[0]) + self.conv(x[1])
            loss = torch.nn.functional.l1_loss(output, t)
            return loss

def test_export_as_training():
    m  = ConvAndLoss()
    sd = {k:v.clone() for k,v in m.state_dict().items()}

    loss_func = torch.nn.functional.l1_loss
    x_        = torch.ones([1,3,10,10])
    x         = [x_, x_]
    t         = torch.zeros([1,1,10,10])
    optim     = torch.optim.SGD(m.parameters(), lr=0.0002, momentum=0.9)
    
    exported  = torchscriptlib.export_model_for_training(m, optim)
    assert not isinstance(exported, Exception), exported
    
    assert state_dicts_allclose(sd, m.state_dict())

    n = 4
    ts_outputs = [ exported.optimizerstate ]
    for i in range(n):
        output = exported.torchscriptmodule(x, t, ts_outputs[-1])
        ts_outputs.append(output)
    
    assert state_dicts_allclose(sd, m.state_dict())
    assert state_dicts_allclose(sd, exported.torchscriptmodule.state_dict())



    eager_outputs:tp.List[tp.Dict] = [{}]
    for i in range(n):
        m.zero_grad()
        loss = m(x, t)
        loss.backward()
        optim.step()
        eager_outputs.append( {'loss':loss.item()} )
    print()

    
    for i,(ts_out, eager_out) in enumerate(zip(ts_outputs, eager_outputs)):
        print(f'----- {i} -----')
        keys = eager_out.keys()
        for k in keys:
            ts_val = np.asarray(ts_out[k].detach())
            ea_val = np.asarray(eager_out[k])
            diff   = np.abs(ts_val - ea_val).max()
            print(k)
            print(f'TorchScript: {ts_val.ravel()[-5:]}')
            print(f'Eager Torch: {ea_val.ravel()[-5:]}')
            print(f'Diff:        {diff}')

            assert np.allclose(ts_val, ea_val)
            print()


    #assert 0


def state_dicts_allclose(sd0, sd1):
    return all([
        torch.allclose(p0,p1) for p0,p1 in zip(sd0.values(), sd1.values())
    ])
