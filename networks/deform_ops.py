import torch
import torch.nn as nn

try:
    # raise Exception("not work")
    from tvdcn.ops import PackedDeformConv2d, PackedDeformConv3d
    print("tvdcn is installed, using it for deformable convolution")
    
    class DeformConv2d(PackedDeformConv2d):
        def __init__(self, in_channels, out_channels, kernel_size, 
                    stride=1, padding=0, dilation=1, groups=1, 
                    offset_groups=1, mask_groups=1, bias=True, 
                    generator_bias: bool = False, 
                    deformable: bool = True, modulated: bool = False):
            super().__init__(in_channels, out_channels, kernel_size, 
                            stride, padding, dilation, groups, offset_groups, 
                            mask_groups, bias, generator_bias, deformable, modulated)

        def forward(self, x):
            return super().forward(x)


    class DeformConv3d(PackedDeformConv3d):
        def __init__(self, in_channels, out_channels, kernel_size, 
                    stride=1, padding=0, dilation=1, groups=1, 
                    offset_groups=1, mask_groups=1, bias=True, 
                    generator_bias: bool = False, 
                    deformable: bool = True, modulated: bool = False):
            super().__init__(in_channels, out_channels, kernel_size, 
                            stride, padding, dilation, groups, offset_groups, 
                            mask_groups, bias, generator_bias, deformable, modulated)

        def forward(self, x):
            return super().forward(x)
except:
    try:
        # raise Exception("not work")
        from mmcv.ops import DeformConv2dPack
        print("tvdcn is not installed, using mmcv for deformable convolution")
        class DeformConv2d(DeformConv2dPack):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):
                x = x.contiguous()
                return super().forward(x)
    except:
        from torchvision.ops import deform_conv2d
        print("Neither tvdcn nor mmcv is not installed, using torchvision for deformable convolution")
        class DeformConv2d(nn.Conv2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.offset_conv = nn.Conv2d(self.in_channels,
                                         2 * self.kernel_size[0] * self.kernel_size[0],
                                         kernel_size=self.kernel_size,
                                         stride=self.stride,
                                         padding=self.padding)

                nn.init.constant_(self.offset_conv.weight, 0.)
                nn.init.constant_(self.offset_conv.bias, 0.)

            def forward(self, x):
                offset = self.offset_conv(x)
                x = deform_conv2d(input=x,
                                 offset=offset,
                                 weight=self.weight,
                                 bias=self.bias,
                                 padding=self.padding,
                                 mask=None,
                                 stride=self.stride,
                                 dilation=self.dilation)
                return x



