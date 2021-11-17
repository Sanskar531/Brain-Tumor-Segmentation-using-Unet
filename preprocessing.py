import mat73
import numpy as np
import SimpleITK as sitk
from scipy.io import savemat
from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf

def normalize_minMax(X):
    """
    Normalize using Min-Max Normalization
    """
    mini = X.min();
    maxi = X.max();
    return (X- mini)/(maxi-mini);


def remove_bias(X):
    """
    Correcting Bias Field using SimpleITK's N4BiasFieldCorrectionImageFilter class.
    """
    result = []
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetConvergenceThreshold(0.01)
    corrector.SetMaximumNumberOfIterations([100,100,100,100])
    for i in range(len(X)):
        print(i)
        img = sitk.GetImageFromArray(np.reshape(X[i], (512,512)))
        img = sitk.Cast(img, sitk.sitkFloat32)
        a = sitk.GetArrayFromImage(corrector.Execute(img))
        result.append(a)
    return result;

def filter_diff(X,y,y_mask, size): 
    """ 
    Some images in the dataset aren't of the shape (512,512).
    This function removes those images.
    
    """
    j=len(X)-1;
    while j>=0:
        if X[j].shape != size:
            del X[j];
            del y[j];
            del y_mask[j];
        else:
            j-=1;

def preprocess_data(data):
    X = [i["image"] for i in data["data"]["cjdata"]];  
    y_mask = [i["tumorMask"] for i in data["data"]["cjdata"]];
    y = [i["label"] for i in data["data"]["cjdata"]];
    # filter out images of different dimensions
    filter_diff(X,y,y_mask, (512,512));
    X = normalize_minMax(np.array(X));# normalize the x
    X = remove_bias(X);# remove bias from X
    #resizing x and y_mask for the input layer in the networks
    X = np.reshape(X,(len(X),X[0].shape[0],X[0].shape[1],1)); 
    # change y from denoting class as a number to a hot encoded binary matrix
    y = keras.utils.to_categorical(np.array(data["y"])-1,3)
    y_mask = np.array(y_mask).astype(int);
    y_mask =np.reshape(y_mask,(len(y_mask),y_mask[0].shape[0],y_mask[0].shape[1],1));
    y_mask = tf.image.resize(y_mask, (324,324)).numpy();
    # saving the file so it can be shared
    savemat("./N4Corrected2.mat", {"X":X, "y":y, "y_mask":y_mask})