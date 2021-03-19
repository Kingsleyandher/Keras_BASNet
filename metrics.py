import keras.backend as K
from keras import layers
from keras.losses import binary_crossentropy
from skimage.metrics import structural_similarity
import tensorflow as tf

def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(tf.cast(y_true, tf.float64), tf.cast(y_pred, tf.float64), 1.0))

def bce_ssim_loss(y_true, y_pred):
    bce_out = binary_crossentropy(y_true,y_pred)
    ssim_out = 1 - ssim(y_true,y_pred)
    iou_out = 1 - compute_IOU(y_true,y_pred)
    loss = bce_out + ssim_out + iou_out
    return loss

def output_loss(y_true, y_pred):
    """
    返回八张特征图最接近图片边缘的那一张的bce-ssim-loss
    """
    targets = tf.transpose(y_true[...,0],[1,2,0])
    predict = tf.transpose(y_pred[...,0],[1,2,0])
    loss0 = bce_ssim_loss(targets,predict)
    return loss0

def muti_bce_loss_fusion(y_true, y_pred):
    #y_pred: batch_size,256,256,8
    loss = bce_ssim_loss(y_true, y_pred)
    return loss

def compute_IOU(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def dice_coef(y_true, y_pred):
    """
    该函数可以求得 2*∩+1/（∪+1）
    用来评估分割结果好坏的函数，值越大代表分割效果越好。
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
