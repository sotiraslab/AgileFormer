import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume, calculate_metric_list_percase
from optimizer_factory import build_optimizer
from lr_scheduler_factory import build_scheduler
from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator, ValGenerator


def inference(args, model, testloader, test_save_path=None):
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(testloader.dataset)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

def validation(model, validloader, classes=9):
    # mask_pred, mask_gt = [], []
    model.eval()
    running_metric = 0.
    with torch.no_grad():
        for sampled_batch in validloader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print(image_batch.shape)
            bsz = image_batch.shape[0]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            prob_batch = torch.softmax(outputs, dim=1)
            pred_batch = prob_batch.argmax(dim=1).cpu()

            label_batch_np = label_batch.squeeze().cpu().numpy()
            pred_batch_np = pred_batch.squeeze().cpu().numpy()

            running_metric += calculate_metric_list_percase(pred_batch_np, label_batch_np, classes, False) * bsz

    running_metric /= len(validloader.dataset)

    return running_metric

def trainer_ACDC(args, model, snapshot_path):
    # from datasets.dataset_synapse import Synapse_dataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid",
                               transform=transforms.Compose(
                                   [ValGenerator(output_size=[args.img_size, args.img_size])]))
    db_test = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test",
                               transform=None)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = build_optimizer(args, model)
    if not args.lr_scheduler in ['const', 'exponential']:
        lr_scheduler = build_scheduler(args, optimizer, len(trainloader))
    # writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        epoch_loss = 0.0
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = (1 - args.dice_loss_weight) * loss_ce + args.dice_loss_weight * loss_dice
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            if args.lr_scheduler == 'exponential':
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            elif args.lr_scheduler == 'const':
                lr_ = base_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_scheduler.step_update(epoch_num * len(trainloader) + i_batch)

            epoch_loss += loss.item()
            iter_num = iter_num + 1

            if iter_num % 20 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, mem: %.0fMB' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), memory_used))

        # Testing 
        epoch_loss /= len(trainloader)
        metric_valid_list = validation(model, validloader, num_classes)
        metric_valid_mean = np.mean(metric_valid_list, axis=0)[0]
        logging.info('epoch %d : mean dice: %f' % (epoch_num, metric_valid_mean))

        if metric_valid_mean > best_performance:
            best_performance = metric_valid_mean
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Best Model Saved at Epoch {epoch_num}!")
            for i in range(1, args.num_classes):
                logging.info('Mean class %d mean_dice %f' % (i, metric_valid_list[i-1][0]))
            logging.info('Testing performance in best val model: mean_dice : %f' % (best_performance))

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            # inference(args, model, testloader, None)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            inference(args, model, testloader, None)

            ## retrive the best model and run full testing
            ckpt = torch.load(os.path.join(snapshot_path, 'best_model.pth'))
            model.load_state_dict(ckpt)
            inference(args, model, testloader, None)

            iterator.close()
            break
    
    return "Training Finished!"