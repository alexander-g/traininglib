from traininglib.unet import UNet
import torch


def test_unet():
    #test multiple input channels
    in_c  = 7
    out_c = 2
    model = UNet(input_channels = in_c, output_channels = out_c, backbone_weights = None)

    x     = torch.ones([2, in_c, 128, 128])
    y     = model(x)
    assert y.shape == (2, out_c, 128, 128)

