import torch





def test_warp_tensor_with_flow() -> None:
    from traininglib.flow import warp_tensor_with_flow

    h,w  = 5,4
    x    = torch.ones([3,h,w])

    flow = torch.ones([2,h,w], requires_grad=True)

    result = warp_tensor_with_flow(x[None], flow[None] * 2, 'bilinear')[0]

    # make sure that the operation is differentiable
    (result.sum()).backward()
    assert flow.grad is not None

    result = result.detach()
    assert np.allclose( result[:, :, 0], 0 )
    assert np.allclose( result[:, 0, :], 0 )
    assert np.allclose( result[:, -1, -1], 1 )
