import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'AgileFormer'
# Model name
_C.MODEL.NAME = 'AgileFormer-Tiny'
_C.MODEL.PRETRAIN_CKPT = './pretrained_ckpt/dat_pp_tiny_in1k_224.pth'
_C.MODEL.Params = CN(new_allowed=True)

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    # config.defrost()
    # if args.opts:
    #     config.merge_from_list(args.opts)

    # # merge from specific arguments
    # if args.batch_size:
    #     config.DATA.BATCH_SIZE = args.batch_size
    # if args.data_path:
    #     config.DATA.DATA_PATH = args.data_path
    # if args.resume:
    #     config.MODEL.RESUME = args.resume
    # if args.amp:
    #     config.AMP = args.amp
    # if args.print_freq:
    #     config.PRINT_FREQ = args.print_freq
    # if args.output:
    #     config.OUTPUT = args.output
    # if args.tag:
    #     config.TAG = args.tag
    # if args.eval:
    #     config.EVAL_MODE = True
    # if args.throughput:
    #     config.THROUGHPUT_MODE = True

    # # output folder
    # config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    # config.freeze()


def get_config(cfg_file):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, cfg_file)
    _update_config_from_file(config, cfg_file)

    return config