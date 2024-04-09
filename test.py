import os
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_ACDC import ACDC_dataset
from networks.agileFormer_2d import AgileFormer2D as ViT_seg
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse', help='root dir for data')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='dataset_name')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", 
                    help='path to config file', )
parser.add_argument('--save_dir', type=str,
                    default='./predictions', help='save directory of testing')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
args = parser.parse_args()
print(args)

def inference(args, model, test_save_path=None):
    dataset_config = args.dataset_config
    if args.dataset == 'Synapse':
        db_test = dataset_config[args.dataset]['Dataset'](base_dir=join(args.volume_path, "test_vol_h5"), split="test_vol", list_dir=args.list_dir)
    elif args.dataset == 'ACDC':
        db_test = dataset_config[args.dataset]['Dataset'](base_dir=args.volume_path, split="test", list_dir=args.list_dir)
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    args.logger.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=dataset_config[args.dataset]['z_spacing'])
        metric_list += np.array(metric_i)
        args.logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(testloader.dataset)
    for i in range(1, args.num_classes):
        args.logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    args.logger.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))


if __name__ == "__main__":
    save_dir = join(args.save_dir, args.dataset)
    maybe_mkdir_p(save_dir)
    args.logger = get_logger(join(save_dir, "test_info.log"))
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    dataset_name = args.dataset
    dataset_config = {
            'Synapse': {
                'Dataset': Synapse_dataset,
                'volume_path': args.volume_path,
                'list_dir': args.list_dir,
                'num_classes': args.num_classes,
                'z_spacing': 1,
            },
            'ACDC': {
                'Dataset': ACDC_dataset,
                'volume_path': args.volume_path,
                'list_dir': args.list_dir,
                'num_classes': args.num_classes,
                'z_spacing': 1,
            },
        }
    args.dataset_config = dataset_config

    args.img_size = 224
    config = get_config(args.cfg)

    net = ViT_seg(config, num_classes=args.num_classes).cuda()

    msg = net.load_state_dict(torch.load(config.MODEL.PRETRAIN_CKPT))
    args.logger.info(msg)

    # args.logger.info("Testing Started ...")
    # inference(args, net, save_dir)
    # args.logger.info("Testing Finished !")