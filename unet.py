import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Conv2D, Lambda, Input, UpSampling2D 
from tensorflow.compat.v1.keras.layers import MaxPool2D, Dropout, concatenate, BatchNormalization
from tensorflow.compat.v1.keras.layers import Activation, Conv2DTranspose, Concatenate
from tensorflow.compat.v1 import keras

def sensitivity( y_true, y_pred):
    """
            True Postive
    -----------------------------
    True Postive + False Negatives
    """
    
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    AP = K.sum(K.round(K.clip(y_true, 0, 1)))
    return TP / (AP + K.epsilon()) # epsilon added to aviod divide by 0 error.

def specificity( y_true, y_pred):
    """
            True Negative
    -----------------------------
    True Negative + False Positives
    """
    
    TN = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    AN = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return TN / (AN + K.epsilon()) # epsilon added to aviod divide by 0 error.

def dice_score( y_true, y_pred):
    """
    Calcutes the amount of overlapping two images have
    """
    flat_true = K.flatten(y_true)
    flat_pred = K.flatten(y_pred)
    intersection = K.sum(flat_true * flat_pred)
    return (2. * intersection + K.epsilon()) / (
                K.sum(flat_true) + K.sum(flat_pred) + K.epsilon())

def loss_dice( y_true, y_pred):
    """
    1 minus dice score gives us a number between 0 and 1. When score is low, cost is high, 
    whereas, when score is high loss is low.
    This was used to train/penalize the model based on the coefficient.
    """
    loss = 1 - dice_score(y_true, y_pred)
    return loss

def loss_WCE(y_true, y_pred):
    """
     Weighted cross entropy used to give more weight to false negatives using beta value 
     greater than 1. Here, beta is denoted
     by the constant beta.
    """
    beta = 30;
    y_trueFl32=tf.cast(y_true, tf.float32);
    y_predFl32=tf.cast(y_true, tf.float32);
    weight_a = beta * y_trueFl32;
    weight_b = 1 - y_predFl32;

    loss = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b 
    return tf.reduce_mean(loss);

def unet(input_size) : #follows the U-net Architecture
    inputs = Input(input_size)
    c1 = Conv2D(64, 3, activation = 'relu')(inputs)
    c1 = Conv2D(64, 3, activation = 'relu')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPool2D(pool_size=(2, 2))(c1)
    c2 = Conv2D(128, 3, activation = 'relu')(p1)
    c2 = Conv2D(128, 3, activation = 'relu')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPool2D(pool_size=(2, 2))(c2)
    c3 = Conv2D(256, 3, activation = 'relu')(p2)
    c3 = Conv2D(256, 3, activation = 'relu')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPool2D(pool_size=(2, 2))(c3)
    c4 = Conv2D(512, 3, activation = 'relu')(p3)
    c4 = Conv2D(512, 3, activation = 'relu')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPool2D(pool_size=(2, 2))(c4)

    conv5 = Conv2D(1024, 3, activation = 'relu')(p4)
    conv5 = Conv2D(1024, 3, activation = 'relu')(conv5)

    T6 = Conv2DTranspose(1024, 2, activation = 'relu', strides=(2,2))(conv5)
    crop4 = tf.image.resize(c4, (T6.shape[1],T6.shape[2]))
    merge7 = concatenate([crop4,T6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu')(T6)
    conv6 = Conv2D(512, 3, activation = 'relu')(conv6)
    conv6 = BatchNormalization()(conv6)

    T7 = Conv2DTranspose(512, 2, activation = 'relu',strides=(2,2))(conv6)
    crop3 = tf.image.resize(c3, (T7.shape[1],T7.shape[2]))
    merge7 = concatenate([crop3,T7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu')(conv7)
    conv7 = BatchNormalization()(conv7)

    T8 = Conv2DTranspose(256, 2, activation = 'relu',strides=(2,2))(conv7)
    crop2 = tf.image.resize(c2, (T8.shape[1],T8.shape[2]))
    merge8 = concatenate([crop2,T8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu')(conv8)
    conv8 = BatchNormalization()(conv8)

    T9 = Conv2DTranspose(128, 2, activation = 'relu',strides=(2,2))(conv8)
    crop1 = tf.image.resize(c1, (T9.shape[1],T9.shape[2]))
    merge9 = concatenate([crop1,T9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    c10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = keras.Model(inputs = inputs, outputs = c10)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss = loss_WCE, metrics = ['accuracy', dice_score, sensitivity, specificity])

    
    return model