# AgileFormer
This official implementation for the paper titled "AgileFormer: Spatially Agile Transformer UNet for Medical Image Segmentation" (https://arxiv.org/abs/2404.00122)

![Method](figures/cover.jpg)

**Abstract.** In the past decades, deep neural networks, particularly convolutional neural networks, have achieved state-of-the-art performance in a variety of medical image segmentation tasks. Recently, the introduction of the vision transformer (ViT) has significantly altered the landscape of deep segmentation models. There has been a growing focus on ViTs, driven by their excellent performance and scalability. However, we argue that the current design of the vision transformer-based UNet (ViT-UNet) segmentation models may not effectively handle the heterogeneous appearance (e.g., varying shapes and sizes) of objects of interest in medical image segmentation tasks. To tackle this challenge, we present a structured approach to introduce spatially dynamic components to the ViT-UNet. This adaptation enables the model to effectively capture features of target objects with diverse appearances. This is achieved by three main components: (i) deformable patch embedding; (ii) spatially dynamic multi-head attention; (iii) deformable positional encoding. These components were integrated into a novel architecture, termed AgileFormer. AgileFormer is a spatially agile ViT-UNet designed for medical image segmentation. Experiments in three segmentation tasks using publicly available datasets demonstrated the effectiveness of the proposed method

## 1. Prepare data

- [Synapse multi-organ segmentation] The Synapse datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). 
- [ACDC cardiac segmentation]
- [Decathlon brain tumor segmentation]

## 2. Environment
- We recommend an evironment with python >= 3.8, and then install the following dependencies:
```
pip install -r requirements.txt
```

- We recommend to install **Neighborhood Attention (NATTEN)** and **Defomrable Convolution** manually for compatability issues:
    - [**NATTEN**] Please refer to https://shi-labs.com/natten to install NATTEN with correct CUDA and PyTorch versions (**Note: we trained the model using CUDA 12.1 + PyTorch 2.2, and NATTEN=0.15.1**). 
    For example, we can install NATTEN with Pytorch 2.2 and CUDA 12.1 with 
    ```
    pip3 install natten==0.15.1+torch220cu121 -f https://shi-labs.com/natten/wheels/
    ```
    - [Deformable Convolution] There are many implementation of deformable convolution:
        - [**tvdcn**] We recommend the implementation in **tvdcn** (https://github.com/inspiros/tvdcn), as it provides CUDA implementation of both 2D/3D deformable convolution (The 2D implementation of deformable convolution in tvdcn should be the same as that provided by PyTorch) [**Note: We used tvdcn for our experiments**]
        For example, we can install latest tvdcn with Pytorch >= 2.1 and CUDA >= 12.1 with
        ```
        pip install tvdcn
        ```
        - [**mmcv**] We also provide an alternative implementaiton of deformable convolution in mmcv (https://github.com/open-mmlab/mmcv). This is the most widely used version; but it only provides 2D CUDA implementation.
        The installation of mmcv is quite straightforward with (you may need to check PyTorch and CUDA version as well)
        ```
        pip install -U openmim 
        mim install mmcv
        ```
        - [**vanilla PyTorch**] We also provide the implementation provided by official PyTorch
        - **Note:** Our code will search all the aforementioned three options in order: if tvdcn is installed, we will use it; elif mmcv is installed, we will use mmcv; else we will use implementation provided by Pytorch.

- **Final Takeaway:** We suggest installing PyTorch >= 2.1, CUDA >= 12.1 for better compatability of all pacakges (especially tvdcn and natten). It is also possible to install those two packages with lower PyTorch and CUDA version, but may need to build from source. 

## 3. Evaluate Pretrained Models 
We provide the pretrained models in the tiny and base versions of AgileFormer, as listed below.

| task  | model size | resolution | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: |
| Synapse multi-organ segmentation (2D) | Tiny | 224x224 | [config](configs/agileFormer_lite.yaml) | |
| Synapse multi-organ segmentation (2D) | Base | 224x224 | [config](configs/agileFormer_base.yaml) | [Google Drive](https://drive.google.com/file/d/1yGPZ2M2zJGhl-FVeQ9dGAYG9G_swNCAC/view?usp=sharing) / [One Drive](https://gowustl-my.sharepoint.com/:u:/r/personal/peijie_qiu_wustl_edu/Documents/AgileFormer_pretrained_ckpt/model_base_w_DS.pth?csf=1&web=1&e=1vAej1) |
| ACDC cardiac segmentation (2D) | Tiny | 224x224 | [config](configs/agileFormer_lite.yaml) | |
| ACDC cardiac segmentation (2D) | Base | 224x224 | [config](configs/agileFormer_base.yaml) | |
| Decathlon brain tumor segmentation (3D) | Tiny | 96x96x96 | [config](configs/agileFormer_lite.yaml) | |
| Decathlon brain tumor segmentation (3D) | Base | 96x96x96 | [config](configs/agileFormer_base.yaml) | |

Put pretrained weights into folder **"pretrained_ckpt/"** under the main "AgileFormer" directory
```
python test.py
```

## 4. Train From Scratch

### a. Download pre-trained deformable attention weights (DAT++)
| model  | resolution | pretrained weights |
| :---: | :---: | :---: | 
| Tiny | 224x224 | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgrl-pI8MPFoll-ueNQ?e=bpdieu) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/14c5ddae10b642e68089/) |
| Base | 224x224 | [OneDrive](https://1drv.ms/u/s!ApI0vb6wPqmtgrl_P46QOehhgA0-wg?e=DJRAfw) / [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/8e30492404d348d89f25/) |

If you are interested in more pretrained weights (e.g., with different resolutions, model sizes, and tasks), please check with the official repo in DAT++: (https://github.com/LeapLabTHU/DAT)

Put pretrained weights into folder **"pretrained_ckpt/"** under the main "AgileFormer" directory

### b. Run the training script
```
python train.py
```

## Future Updates
- [x] Release the tentative code for 2D segmentation.
- [x] Release the pretrained code for 2D segmentation.
- [ ] Reorganize the tentative code for easier usage.
- [ ] Release the code for 3D segmentation.
- [ ] Release the pretrained code for 3D segmentation.
- [ ] Support mmcv and pytorch deformable convolution

## Acknowledgements

This code is developed on the top of [Swin UNet](https://github.com/HuCaoFighting/Swin-Unet) and [DAT](https://github.com/LeapLabTHU/DAT), we thank to their efficient and neat codebase. 

## Citation
If you find our work is useful in your research, please consider citing:

```
@article{qiu2024agileformer,
  title={AgileFormer: Spatially Agile Transformer UNet for Medical Image Segmentation},
  author={Qiu, Peijie and Yang, Jin and Kumar, Sayantan and Ghosh, Soumyendu Sekhar and Sotiras, Aristeidis},
  journal={arXiv preprint arXiv:2404.00122},
  year={2024}
}
```