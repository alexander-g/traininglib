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


def test_slice_mask():
    slice_mask = torch.jit.script(skel.slice_mask)
    patch, topleft = slice_mask(A.bool())
    assert torch.all(patch == A[:,1:-2])
    assert topleft.numpy().tolist() == [0,1]

    patch2, topleft2 = slice_mask( (A*0).bool() )
    assert patch2.shape == (0,0)
    assert torch.all(topleft2 == 0)



#how many channels the output should contain (currently: x,y. maybe more later)
EXPECTED_DIM2 = 2

def test_skel2path_invalid():
    # dont raise exception
    out = skel.path_via_dfs(~A.bool())
    assert out.shape == (0,EXPECTED_DIM2)

def test_skel2path_valid():
    dfs  = torch.jit.script(skel.path_via_dfs)
    path = dfs(A.bool())
    assert path.shape == (8,EXPECTED_DIM2)
    assert [0,1] == path.tolist()[0]
    assert [6,4] == path.tolist()[-1]

    session  = _export_to_onnx(dfs, args=(A.bool(),), dyn_ax={'x':[0,1]})
    onnx_out = session.run(None, {'x':A.bool().numpy()})[0]
    
    assert onnx_out.shape == (8,EXPECTED_DIM2)
    assert [0,1] == onnx_out.tolist()[0]
    assert [6,4] == onnx_out.tolist()[-1]

    
    


def test_multipath():
    skel_f = torch.jit.script(skel.paths_from_labeled_skeleton)
    A2 = torch.cat([A,A,A], dim=1)
    L  = torch.cat([
        torch.ones_like(A)*5, 
        torch.ones_like(A)*7,
        torch.ones_like(A)*65,
    ], dim=1)
    labeled_A = A2*L

    labeled_paths = skel_f(labeled_A)
    print(labeled_paths)
    assert labeled_paths.shape == (8*3, EXPECTED_DIM2+1)
    assert labeled_paths[::8,2].tolist() == [5,7,65]
    #make sure the coordinates are somewhat correct
    assert torch.all(labeled_paths[:8,1] < 6)
    assert torch.all(labeled_paths[8:16,1] > 6)
    assert torch.all(labeled_paths[8:16,1] < 12)
    assert torch.all(labeled_paths[16:,1]  > 12)

