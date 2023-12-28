import typing as tp
import torch
import numpy as np

from traininglib import torchscriptlib

TensorDict = tp.Dict[str, torch.Tensor]

class ConvBNAndLoss(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(3,3, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(3),
                torch.nn.Conv2d(3,1, kernel_size=1, padding=0),
            )

        def forward(self, inputs:TensorDict) -> tp.Tuple[torch.Tensor, TensorDict]:
            x0 = inputs['x0']
            x1 = inputs['x1']
            t  = inputs['t']
            output = self.seq(x0) + self.seq(x1)
            loss = torch.nn.functional.l1_loss(output, t)
            return loss, {}

def test_export_as_training():
    m  = ConvBNAndLoss()
    sd = {k:v.clone() for k,v in m.state_dict().items()}

    loss_func = torch.nn.functional.l1_loss
    x         = torch.ones([1,3,10,10])
    t         = torch.zeros([1,1,10,10])
    optim     = torch.optim.SGD(m.parameters(), lr=0.0002, momentum=0.9)
    
    exported  = torchscriptlib.export_model_for_training(m, optim)
    assert not isinstance(exported, Exception), exported
    
    assert state_dicts_allclose(sd, m.state_dict())

    print(exported.torchscriptmodule.code)

    n = 4
    ts_outputs = [ exported.trainingstate ]
    for i in range(n):
        inputfeed = ts_outputs[-1] | {'x0':x, 'x1':x, 't':t}
        output    = exported.torchscriptmodule(inputfeed)
        ts_outputs.append(output)
        print()
    
    assert state_dicts_allclose(sd, m.state_dict())
    assert state_dicts_allclose(sd, exported.torchscriptmodule.state_dict())



    eager_outputs:tp.List[tp.Dict] = [{}]
    for i in range(n):
        m.zero_grad()
        loss,_ = m({'x0':x, 'x1':x,  't':t})
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


    #exported.torchscriptmodule.save('DELETE.torchscript')
    #assert 0


def state_dicts_allclose(sd0, sd1):
    return all([
        torch.allclose(p0,p1) for p0,p1 in zip(sd0.values(), sd1.values())
    ])
