import typing as tp
import torch

from traininglib import torchscriptlib



def test_export_as_training():
    m = torch.nn.Sequential(
        torch.nn.Conv2d(3,5, kernel_size=3, padding=1),
        torch.nn.Conv2d(5,1, kernel_size=1),
    )
    class M(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.conv = torch.nn.Conv2d(3,1, kernel_size=3, padding=1)

        def forward(self, x:tp.List[torch.Tensor], t:torch.Tensor):
            output = self.conv(x[0]) + self.conv(x[1])
            loss = torch.nn.functional.l1_loss(output, t)
            return loss

        def _forward(self, x:tp.Dict[str, torch.Tensor]):
            return self.conv(x['0']) + self.conv(x['1'])
    m = M()

    loss_func = torch.nn.functional.l1_loss
    x         = torch.ones([1,3,10,10])
    x = [x,x]
    t         = torch.zeros([1,1,10,10])
    optim     = torch.optim.SGD(m.parameters(), lr=0.0002, momentum=0.9)

    exported  = torchscriptlib.export_model_for_training(
        m, loss_func, x, t, optim
    )
    assert not isinstance(exported, Exception), exported

    output = exported.torchscriptmodule(x, t, exported.optimizerstate)
    print(output.keys())

    for i in range(10):
        output = exported.torchscriptmodule(x, t, output)
        print(output['loss'].item())
        print()

    #assert 0

