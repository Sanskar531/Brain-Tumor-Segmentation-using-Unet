import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf

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