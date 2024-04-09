# AgileFormer
This official repo for the paper titled "AgileFormer: Spatially Agile Transformer UNet for Medical Image Segmentation" (https://arxiv.org/abs/2404.00122)



## 1. Download pre-trained deformable attention weights (DAT++)
| model  | resolution | pretrained weights |
| :---: | :---: | :---: | 
| Tiny | 224x224 | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgrl-pI8MPFoll-ueNQ?e=bpdieu) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/14c5ddae10b642e68089/) |
| Base | 224x224 | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgrl_P46QOehhgA0-wg?e=DJRAfw) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8e30492404d348d89f25/) |

If you are interested in more pretrained weights (e.g., with different resolutions, model sizes, and tasks), please check with the official repo in DAT++: (https://github.com/LeapLabTHU/DAT)

Put pretrained weights into folder **"pretrained_ckpt/"** under the main "AgileFormer" directory

## 2. Prepare data

- [Synapse multi-organ segmentation] The Synapse datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). 
- [ACDC cardiac segmentation]
- [Decathlon brain tumor segmentation]

## 3. Environment
- We recommend an evironment with python >= 3.8, and then install the following dependencies:
```
pip install -r requirements.txt
```

- We recommend to install **Neighborhood Attention (NATTEN)** and **Defomrable Convolution** manually for compatability issues:
    - [NATTEN] Please refer to https://shi-labs.com/natten to install NATTEN with correct CUDA and PyTorch versions (**Note: we trained the model using CUDA 12.1 + PyTorch 2.2, and NATTEN=0.15.1**)
    - [Deformable Convolution] There are many implementation of deformable convolution:
        - [tvdcn] We recommend the implementation in **tvdcn** (https://github.com/inspiros/tvdcn), as it provides CUDA implementation of both 2D/3D deformable convolution (The 2D implementation of deformable convolution in tvdcn should be the same as that provided by PyTorch) [**Note: We used tvdcn for our experiments**]
        - [mmcv] We also provide an alternative implementaiton of deformable convolution in mmcv (https://github.com/open-mmlab/mmcv). This is the most widely used version; but it only provides 2D CUDA implementation.
        - [vanilla PyTorch] We also provide the implementation provided by official PyTorch
        - **Note:** Our code will search all the aforementioned three options in order: if tvdcn is installed, we will use it; elif mmcv is installed, we will use mmcv; else we will use implementation provided by Pytorch.

- **Final Takeaway:** 

## 4. Evaluate Pretrained Models 
We provide the pretrained models in the tiny and base versions of AgileFormer, as listed below.

| task  | model size | resolution | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: |
| Synapse multi-organ segmentation (2D) | Tiny | 224x224 | [config](configs/agileFormer_lite.yaml) | |
| Synapse multi-organ segmentation (2D) | Base | 224x224 | [config](configs/agileFormer_base.yaml) | |
| ACDC cardiac segmentation (2D) | Tiny | 224x224 | [config](configs/agileFormer_lite.yaml) | |
| ACDC cardiac segmentation (2D) | Base | 224x224 | [config](configs/agileFormer_base.yaml) | |
| Decathlon brain tumor segmentation (3D) | Tiny | 96x96x96 | [config](configs/agileFormer_lite.yaml) | |
| Decathlon brain tumor segmentation (3D) | Base | 96x96x96 | [config](configs/agileFormer_base.yaml) | |

Put pretrained weights into folder **"pretrained_ckpt/"** under the main "AgileFormer" directory
```
python test.py
```

## 5. Train from scratch