# Focal Tversky Attention U-Net

This repo contains the code accompanying our paper [A novel focal Tversky loss function and improved Attention U-Net for lesion segmentation](https://arxiv.org/abs/1810.07842) accepted at [ISBI 2019](https://biomedicalimaging.org/2019/).

Figure 1 in the paper is parametrized by the function ![](https://latex.codecogs.com/gif.latex?1%20-%28TI_c%29%5E%7B%7B%5Cfrac%7B1%7D%7B%5Cgamma%7D%7D%7D)

However, the code in this repository follows the parametrization: ![](https://latex.codecogs.com/gif.latex?%281%20-TI_c%29%5E%7B%7B%5Cfrac%7B1%7D%7B%5Cgamma%7D%7D%7D). I apologize for any confusion! Both parametrizations have the same effect however I found the latter one to be more stable and that is the loss function presented in this repo. 

![Observe the behavior of the loss function with different modulations by gamma](\images\ftl.png)

We utilize attention gating in this repo which follows from [Ozan Oktan and his collaborators](https://arxiv.org/abs/1804.03999). The workflow is depicted below: ![](\images\ag.png)

If you find this code useful, please consider citing our work:

```
@article{focal-unet,
  title={A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}
```

Training files for the ISIC2018 and BUS2017 Dataset B have been added. 
If training with ISIC2018, create 4 folders: `orig_raw` (not used in this code), `orig_gt`, `resized-train`, `resized-gt`, for full 
resolution input images, ground truth and resized images at `192x256` resolution, respectively.

If training with BUS2017, create 2 folders: `original` and `gt` for input data and ground truth data. In the `bus_train.py` script, images 
will be resampled to `128x128` resolution. 


