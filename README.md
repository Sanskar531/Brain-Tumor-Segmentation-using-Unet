# Brain Tumor Segmentation using Unet
**Data**
---
The data we got is from the following link: 
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=7953679

This data consists of 3064 images along with the type of tumors and their tumor mask. Since, each of these images are individual 
.mat files, they are joined together using matlab and then exported as "./stacked.mat".

Using the unet, we segmented those images using their true mask as leverage for the network to learn where the tumors are.  