import os
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from medpy import metric

import matplotlib.pyplot as plt
from matplotlib import colors as C
import matplotlib.colors as mcolors
from segmentation_mask_overlay import overlay_masks

join = os.path.join

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def subfiles(folder: str, join: bool = True, prefix= None,
             suffix = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y

    if prefix is not None and isinstance(prefix, str):
        prefix = [prefix]
    if suffix is not None and isinstance(suffix, str):
        suffix = [suffix]

    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or any([i.startswith(j) for j in prefix]))
           and (suffix is None or any([i.endswith(j) for j in suffix]))]

    if sort:
        res.sort()
    return res

def calculate_metric(pred, gt, hd95=False):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # print(pred.sum(), gt.sum())
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if hd95:
            hd95 = metric.binary.hd95(pred, gt)
        else:
            hd95 = 0
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def calculate_metric_multiclass(pred, gt, classes=8):
    metric_list = []
    for i in range(1, classes): 
        metric_list.append(calculate_metric(pred == i, gt == i))
    return np.array(metric_list)

def prepare_mask(mask, classes):
    masks = []
    for i in range(1, classes):
        masks.append(mask == i)
    return np.stack(masks, -1)

dataset = "Synapse"
base_dir = join("segm_visualization", dataset)
save_dir = join(base_dir, "visualization")
maybe_mkdir_p(save_dir)

### Synapse ###
classes = 9
alpha = 0.5
beta = 0.7
cmaps = mcolors.CSS4_COLORS
methods = ['AgileFormer', 'MERIT', 'CoTr', 'SDAUT', 'SwinUNet']
color_pattles = ['red','cyan','yellow','forestgreen','blue','purple','darkorange','deeppink']
cmap = {k: cmaps[k] for k in color_pattles[:classes-1]}
colors = []
for k in cmap.keys():
    if k == 'blue':
        colors.append(np.array([47, 85, 151]).reshape(3,) / 255.)
    elif k == 'darkorange':
        colors.append(np.array([197, 90, 17]).reshape(3,) / 255.)
    else:
        colors.append(np.array(C.to_rgb(cmap[k])).reshape(3,))

## iterate all subjects
for gt_path in subfiles(join(base_dir, "segm")):
    case_name = gt_path.split(os.sep)[-1].replace("_gt.nii.gz", "")
    case_dir = join(save_dir, case_name)
    maybe_mkdir_p(case_dir)

    image = nib.load(join(base_dir, "images", case_name + "_img.nii.gz")).get_fdata()
    segm_gt = nib.load(gt_path).get_fdata().astype(int)

    segm_predictions = []
    for m in methods:
        pred = nib.load(join(base_dir, m, case_name + "_pred.nii.gz")).get_fdata()
        segm_predictions.append(pred)
    segm_predictions = np.stack(segm_predictions).astype(int)

    inds = []
    for i in range(segm_gt.shape[-1]):
        tmp = len(np.unique(segm_gt[..., i]))
        if (tmp >= 6 and tmp <=8):
            inds.append(i)
    inds = np.array(inds)

    candiates = []
    for i in inds:
        dices = []
        for j in range(segm_predictions.shape[0]):
            tmp = calculate_metric_multiclass(segm_predictions[j, ..., i], segm_gt[..., i], classes).mean(axis=0)[0]
            dices.append(tmp)
        if np.argmax(dices) == 0:
            candiates.append([i] + dices)

    candiates_df = pd.DataFrame(candiates, columns=["slice"] + methods)
    candiates_df.to_csv(join(case_dir, "dice.csv"), index=False)
    
    for c in candiates:
        s = c[0]
        segm_gt_overlay = overlay_masks(image[..., s], prepare_mask(segm_gt[..., s], classes), colors=colors, alpha=alpha, beta=beta)
        segm_agile_overlay = overlay_masks(image[..., s], prepare_mask(segm_predictions[0, ..., s], classes), colors=colors, alpha=alpha, beta=beta)
        segm_merit_overlay = overlay_masks(image[..., s], prepare_mask(segm_predictions[1, ..., s], classes), colors=colors, alpha=alpha, beta=beta)
        segm_cotr_overlay = overlay_masks(image[..., s], prepare_mask(segm_predictions[2, ..., s], classes), colors=colors, alpha=alpha, beta=beta)
        segm_sdaut_overlay = overlay_masks(image[..., s], prepare_mask(segm_predictions[3, ..., s], classes), colors=colors, alpha=alpha, beta=beta)
        segm_swin_overlay = overlay_masks(image[..., s], prepare_mask(segm_predictions[4, ..., s], classes), colors=colors, alpha=alpha, beta=beta)

        image_grid = np.concatenate([segm_gt_overlay, segm_agile_overlay, segm_merit_overlay, segm_cotr_overlay, segm_sdaut_overlay, segm_swin_overlay], axis=1)
        Image.fromarray(image_grid).save(join(case_dir, f"slice_{s}.jpg"))

