import torch
import torch.nn as nn
from .agileFormer_sys_2d import AgileFormerSys2D


def load_pretrained(ckpt_path, model):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_pretrained(checkpoint['model'])
    # print(msg)
    del checkpoint
    torch.cuda.empty_cache()


class AgileFormer2D(nn.Module):
    def __init__(self, config, num_classes, 
                 deform_patch_embed=True,
                 encoder_pos_layers=[2, 3],
                 decoder_pos_layers=[1, 2],
                 deep_supervision=False):
        super().__init__()

        self.agile_former = AgileFormerSys2D(**config.MODEL.Params, num_classes=num_classes, 
                            deform_patch_embed=deform_patch_embed,
                            encoder_pos_layers=encoder_pos_layers,
                            decoder_pos_layers=decoder_pos_layers,
                            deep_supervision=deep_supervision)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.agile_former(x)
        return logits
    
    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        load_pretrained(pretrained_path, self.agile_former)
