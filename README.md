# focal-tversky-unet

This repo contains the code relating to my paper (under review) posted on https://arxiv.org/pdf/1810.07842.pdf. 
If you find this code useful, please consider citing my paper. 

Training files for the ISIC2018 and BUS2017 Dataset B have been added. 
If training with ISIC2018, create 4 folders: `orig_raw` (not used in this code), `orig_gt`, `resized-train`, `resized-gt`, for full 
resolution input images, ground truth and resized images at `192x256` resolution.

If training with BUS2017, create 2 folders: `original` and `gt` for input data and ground truth data. In the `bus_train.py` script, images 
will be resampled to `128x128` resolution. 
