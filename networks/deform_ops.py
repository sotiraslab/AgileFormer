from tvdcn.ops import PackedDeformConv2d, PackedDeformConv3d


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


