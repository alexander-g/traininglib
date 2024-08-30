import os, sys, time
import typing as tp
import numpy as np
import torch, torchvision
from torchvision.models._utils import IntermediateLayerGetter

from .datalib import resize_tensor

class UNet(torch.nn.Module):
    '''Backboned U-Net'''

    class UpBlock(torch.nn.Module):
        def __init__(self, in_c, out_c, inter_c=None):
            #super().__init__()
            torch.nn.Module.__init__(self)
            inter_c        = inter_c or out_c
            self.conv1x1   = torch.nn.Conv2d(in_c, inter_c, 1)
            self.convblock = torch.nn.Sequential(
                torch.nn.Conv2d(inter_c, out_c, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_c),
                #torch.nn.ReLU(),
            )
        def forward(self, x:torch.Tensor, skip_x:torch.Tensor, relu:bool=True) -> torch.Tensor:
            #x = resize_tensor(x, skip_x.shape[-2:], mode='nearest')   #TODO? mode='bilinear
            x = torch.nn.functional.interpolate(x, skip_x.shape[-2:], mode='nearest')
            x = torch.cat([x, skip_x], dim=1)
            x = self.conv1x1(x)
            x = self.convblock(x)
            if relu:
                x  = torch.nn.functional.relu(x)
            return x
    
    def __init__(
        self, 
        backbone                  = 'mobilenet3l', 
        input_channels:int        = 3, 
        output_channels:int       = 1, 
        backbone_weights:str|None = 'DEFAULT'
    ):
        torch.nn.Module.__init__(self)
        factory_func = BACKBONES.get(backbone, None)
        if factory_func is None:
            raise NotImplementedError(backbone)
        self.backbone, C = factory_func(backbone_weights, input_channels)
        self.backbone_name = backbone
        
        self.up0 = self.UpBlock(C[-1]    + C[-2],  C[-2])
        self.up1 = self.UpBlock(C[-2]    + C[-3],  C[-3])
        self.up2 = self.UpBlock(C[-3]    + C[-4],  C[-4])
        self.up3 = self.UpBlock(C[-4]    + C[-5],  C[-5])
        self.up4 = self.UpBlock(C[-5]    + input_channels, 32)
        self.cls = torch.nn.Conv2d(32, output_channels, 3, padding=1)

    def _forward_unet(
        self, 
        x:               torch.Tensor, 
        sigmoid:         bool = False, 
        return_features: bool = False,
    ) -> torch.Tensor:
        device = self.cls.weight.device
        x      = x.to(device)
        
        X = self.backbone(x)
        X = ([x] + [X[f'out{i}'] for i in range(5)])[::-1]
        x = X.pop(0)
        x = self.up0(x, X[0])
        x = self.up1(x, X[1])
        x = self.up2(x, X[2])
        x = self.up3(x, X[3])
        x = self.up4(x, X[4], relu=False)
        if return_features:
            return x
        x = torch.nn.functional.relu(x)
        x = self.cls(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x
    
    forward = _forward_unet
    


class UNetHead(torch.nn.Sequential):
    def __init__(self, channels:tp.List[int] = [32,32,32,1]):
        modules:tp.List[torch.nn.Module]  = []
        for i,c in enumerate(channels[:-1]):
            modules.append(torch.nn.Conv2d(c, channels[i+1], kernel_size=1))
            if i < len(channels) - 2:
                modules.append(torch.nn.ReLU())
        
        super().__init__(*modules)


def _clone_conv2d_with_new_input_channels(
    prev:torch.nn.Conv2d, new_input_channels:int
) -> torch.nn.Conv2d:
    new_conv = torch.nn.Conv2d(
        in_channels  = new_input_channels,
        out_channels = prev.out_channels,    
        kernel_size  = prev.kernel_size,     # type: ignore [arg-type]
        stride       = prev.stride,          # type: ignore [arg-type]
        padding      = prev.padding,         # type: ignore [arg-type]
        dilation     = prev.dilation,        # type: ignore [arg-type]
        groups       = prev.groups,
        bias         = prev.bias is not None
    )
    
    # copy the weights from the previous layer, matching the available channels
    n_copy_channels = min(new_input_channels, prev.in_channels)
    new_conv.weight.data[:, :n_copy_channels, :, :] = prev.weight.data[:, :n_copy_channels, :, :]
    
    if prev.bias is not None:
        new_conv.bias.data = prev.bias.data # type: ignore [union-attr]
    return new_conv

def resnet18_backbone(
    weights:str|None, input_channels:int = 3
) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.resnet18(weights=weights)
    return_layers = dict(relu='out0', layer1='out1', layer2='out2', layer3='out3', layer4='out4')
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [64, 64, 128, 256, 512]
    if input_channels != 3:
        backbone.conv1 = _clone_conv2d_with_new_input_channels(backbone.conv1, input_channels)
    return backbone, channels

def resnet50_backbone(
    weights:str|None, input_channels:int = 3
) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.resnet50(weights=weights)
    return_layers = dict(relu='out0', layer1='out1', layer2='out2', layer3='out3', layer4='out4')
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [64, 256, 512, 1024, 2048]
    if input_channels != 3:
        backbone.conv1 = _clone_conv2d_with_new_input_channels(backbone.conv1, input_channels)
    return backbone, channels

def mobilenet3l_backbone(
    weights:str|None, input_channels:int = 3
) -> tp.Tuple[torch.nn.Module, tp.List[int]]:
    base = torchvision.models.mobilenet_v3_large(weights=weights).features
    return_layers = {'1':'out0', '3':'out1', '6':'out2', '10':'out3', '16':'out4'}
    backbone = IntermediateLayerGetter(base, return_layers)
    channels = [16, 24, 40, 80, 960]
    if input_channels != 3:
        backbone['0'][0] = _clone_conv2d_with_new_input_channels(backbone['0'][0], input_channels)
    return backbone, channels

BACKBONES = {
    'resnet18':    resnet18_backbone,
    'resnet50':    resnet50_backbone,
    'mobilenet3l': mobilenet3l_backbone,
}
