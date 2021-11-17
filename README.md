# Brain Tumor Segmentation using Unet
**Data**
---
The data we got is from the following link: 
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=7953679

This data consists of 3064 images along with the type of tumors and their tumor mask. Since, each of these images are individual 
.mat files, they are joined together using matlab and then exported as "./stacked.mat".

Using the unet, we segmented those images using their true mask as leverage for the network to learn where the tumors are.  

Different loss functions were used since, the normal binary cross entropy did not cope well with the class imbalance we were having in our tumor masks. Hence, we used other loss function named Weighted cross entropy and dice loss which created much better results. 

Some of the results created by our unet using weighted cross entropy(this was the loss that gave us the highest dice coefficient) were: 

![](https://github.com/Sanskar531/Brain-Tumor-Segmentation-using-Unet/blob/master/Result%20images/SomeResults.png)

Predictions vs Ground Truth 

### Note: The notebook has the same code as the python files. 