import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from .blocks_2d import *
from .deform_ops import DeformConv2d
from .nat_2d import NeighborhoodAttention2D


class LayerScale(nn.Module):
    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)


class TransformerStage(nn.Module):
    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, 
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation, 
                 layer_scale_value, use_lpu, log_cpb):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [ 
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity() 
                for _ in range(2 * depths)
            ]
        )
        self.local_perception_units = nn.ModuleList()
        for _ in range(depths):
            if use_lpu:
                self.local_perception_units.append(nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed))
            else: 
                self.local_perception_units.append(nn.Identity())
       
        for i in range(depths):
            if stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, heads, nat_ksize, attn_drop=attn_drop, proj_drop=proj_drop)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):
        x = self.proj(x)
        for d in range(self.depths):
            
            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class LinearPatchExpand2D(nn.Module):
    def __init__(self, dim, scale_factor=2, norm_layer=LayerNormProxy):
        super().__init__()
        self.dim = dim
        self.scale_factor = scale_factor
        self.output_dim = dim // scale_factor if scale_factor == 2 else dim

        self.expand = nn.Linear(dim, scale_factor*dim if scale_factor==2 else (scale_factor**2)*dim, bias=False) if scale_factor > 1 else nn.Identity()
        self.norm = norm_layer(dim // scale_factor if scale_factor==2 else dim) 

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = x.flatten(2).permute(0, 2, 1)
        x = self.expand(x)
        B, L, C = x.shape
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.scale_factor, p2=self.scale_factor, c=self.output_dim)
        x = x.reshape(B, H*self.scale_factor, W*self.scale_factor, self.output_dim) # BxHxWxC
        x = x.permute(0, 3, 1, 2) # BxCxHxW
        x= self.norm(x)

        return x


class AgileFormerSys2D(nn.Module):
    def __init__(self, img_size=224, patch_size=4, expansion=4, num_classes=9,
                 dim_stem=96, dims=[96, 192, 384, 768], 
                 depths_Encoder=[2, 2, 6, 2], depths_Decoder=[2, 2, 6, 2],
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1],
                 offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 nat_ksizes=[3,3,3,3],
                 layer_scale_values=[-1,-1,-1,-1],
                 use_lpus=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 encoder_pos_layers=[0, 1, 2],
                 decoder_pos_layers=[1, 2, 3],
                 deep_supervision=False,
                 **kwargs):
        super().__init__()
      
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
    
        self.patch_proj = nn.Sequential(
            DeformConv2d(3, dim_stem // 2, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            DeformConv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            DeformConv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        
        ################ encoder ################
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_Encoder))]
        # print(len(dpr))
        self.stages = nn.ModuleList()
        self.deform_pos_encoder = nn.ModuleList()
        use_cpe_encoder = encoder_pos_layers[0] != -1
        
        for i in range(len(depths_Encoder)):
            use_cpe_encoder = use_cpe_encoder and (i in encoder_pos_layers)

            if use_cpe_encoder:
                self.deform_pos_encoder.append(DePE(dims[i], dims[i], conv_op=DeformConv2d, groups=dims[i]))
            else:
                self.deform_pos_encoder.append(nn.Identity())
            
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dims[i], dims[i], depths_Encoder[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths_Encoder[:i]):sum(depths_Encoder[:i + 1])], use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i], ksize_qnas[i], nqs[i],qna_activation,
                    layer_scale_values[i], use_lpus[i], log_cpb[i]
                )
            )
            if i < 3:
                img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.cls_norm = LayerNormProxy(dims[-1]) 

        self.stages_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.deform_pos_decoder = nn.ModuleList()
        self.ds_projs = nn.ModuleList()
        use_cpe_decoder = decoder_pos_layers[0] != -1
   
        for i in range(1, len(depths_Decoder)):
            idx = len(depths_Decoder)-1-i

            use_cpe_decoder = use_cpe_decoder and (i in decoder_pos_layers)

            if use_cpe_decoder:
                self.deform_pos_decoder.append(DePE(dims[idx], dims[idx], conv_op=DeformConv2d, groups=dims[idx]))
            else:
                self.deform_pos_decoder.append(nn.Identity())

            img_size = img_size * 2

            self.stages_up.append(
                TransformerStage(
                    img_size, window_sizes[idx], ns_per_pts[idx],
                    dims[idx], dims[idx], depths_Decoder[idx],
                    stage_spec[idx], groups[idx], use_pes[idx],
                    sr_ratios[idx], heads[idx], heads_q[idx], strides[idx],
                    offset_range_factor[idx],
                    dwc_pes[idx], no_offs[idx], fixed_pes[idx],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths_Decoder[:idx]):sum(depths_Decoder[:idx + 1])], use_dwc_mlps[i],
                    ksizes[idx], nat_ksizes[idx], ksize_qnas[idx], nqs[idx],qna_activation,
                    layer_scale_values[idx], use_lpus[idx], log_cpb[idx]
                )
            )

            if deep_supervision:
                self.ds_projs.append(nn.Sequential(
                    LayerNormProxy(dims[idx]),
                    LinearPatchExpand2D(dims[idx], scale_factor=4),
                    nn.Conv2d(in_channels=dims[idx],out_channels=self.num_classes,kernel_size=1,bias=False)
                ))
            else:
                self.ds_projs.append(nn.Identity())
       
            self.concat_back_dim.append(nn.Conv2d(dims[idx]*2, dims[idx], 1, 1, 0))

        self.up_projs = nn.ModuleList()
        for i in range(len(depths_Decoder)):
            idx = len(depths_Decoder)-1-i
            # print(groups[idx])
            scale_factor = 2 if i < len(depths_Decoder) - 1 else 4
            self.up_projs.append(
                LinearPatchExpand2D(dims[idx], scale_factor=scale_factor),
            )

        self.norm_up= LayerNormProxy(dims[0])
        self.output = nn.Conv2d(in_channels=dims[0],out_channels=self.num_classes,kernel_size=1,bias=False)

        self.lower_lr_kvs = lower_lr_kvs
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict):
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            if "patch_proj" in state_key:
                new_state_dict[state_key] = state_value
            elif "down_projs" in state_key:
                new_state_dict[state_key] = state_value
            elif "cls_norm" in state_key:
                new_state_dict[state_key] = state_value
            elif "stages" in state_key and state_key in self.state_dict().keys():
                if self.state_dict()[state_key].shape == state_value.shape:
                    new_state_dict[state_key] = state_value
                    tmp = state_key.split(".")
                    num_depth = int(tmp[1])
                    if num_depth < 3:
                        new_state_key = ["stages_up", str(2 - num_depth)] + tmp[2:]
                        new_state_key = ".".join(new_state_key)

                        try:
                            if self.state_dict()[new_state_key].shape == state_value.shape:
                                new_state_dict[new_state_key] = state_value
                            else:
                                pass
                        except:
                            pass
                else:
                    pass
            else:
                pass

        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    def forward_encoder(self, x):
        x = self.patch_proj(x)
        # print(x.shape)
        x_downsample = []
        for i in range(len(self.stages)):
            # print(x.shape)
            x_downsample.append(x)
            x = self.deform_pos_encoder[i](x)
            x = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
        x = self.cls_norm(x)
        # print(x.shape)
        return x, x_downsample
    
    def forward_decoder(self, x, x_downsample):
        seg_outputs = []
        for i in range(len(self.stages)):
            if i == 0:
                # print(x.shape)
                x = self.up_projs[i](x)
            else:
                x = torch.cat([x, x_downsample[len(self.stages)-1-i]], 1)
                
                x = self.concat_back_dim[i-1](x)
                x = self.deform_pos_decoder[i-1](x)
                x = self.stages_up[i-1](x)
             
                if self.deep_supervision and (i < 3):
                    seg_outputs.append(self.ds_projs[i-1](x)) 
               
                if i < 3:
                    x = self.up_projs[i](x)
                 
        x = self.norm_up(x)  # B L C
        x = self.up_projs[-1](x)
        x = self.output(x)

        if len(seg_outputs) > 0:
            # print(True)
            x += F.interpolate(F.interpolate(seg_outputs[-2], scale_factor=2, mode='bilinear') + seg_outputs[-1], scale_factor=2, mode='bilinear')
  
        return x
    
    def forward(self, x):
        x, x_downsample = self.forward_encoder(x)
        x = self.forward_decoder(x, x_downsample)
        return x
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table', 'deform_pos_encoder', 'deform_pos_decoder'}