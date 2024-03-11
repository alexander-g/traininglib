import traininglib.segmentation.skeletonization as skel
from util import _export_to_onnx

import torch


def test_skeletonization():
    skeletonize = torch.jit.script(skel.skeletonize)
    
    x = torch.zeros([1,1,100,100])
    x[..., 5:10,   5:90] = 1
    x[..., 15:20, 20:70] = 1

    x_sk = skeletonize(x)

    assert torch.all(x_sk[0,0, :7,   :  ] == 0)
    assert torch.all(x_sk[0,0, 7,   7:87] == 1)
    assert torch.all(x_sk[0,0, 8:10, :  ] == 0)

    session = _export_to_onnx(skeletonize)
    session.run(None, {'x':x.numpy()})



A = torch.as_tensor([
    [0,1,0,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0],
])

def test_endpoints():
    find_endpoints = torch.jit.script(skel.find_endpoints)
    ep = find_endpoints(A).numpy()
    assert ep.shape == (3,2)
    assert [0,1] in ep.tolist()
    assert [0,4] in ep.tolist()
    assert [6,4] in ep.tolist()


#how many channels the output should contain (currently: x,y. maybe more later)
EXPECTED_DIM2 = 2

def test_skel2path_invalid():
    # dont raise exception
    out = skel.path_via_dfs(~A.bool())
    assert out.shape == (0,EXPECTED_DIM2)

def test_skel2path():
    dfs  = torch.jit.script(skel.path_via_dfs)
    path = dfs(A.bool())
    print(path)
    assert path.shape == (8,EXPECTED_DIM2)
    assert [0,1] == path.tolist()[0]
    assert [6,4] == path.tolist()[-1]

