
import numpy as np 


import tensorflow as tf

import keras

import keras as keras

from keras.models import Sequential, Model
from keras.layers import *

from keras import backend as K



from frm_dataset_loader import to_categorical  
import os
import sys



pi = np.pi


def rotate(xy):
    x = xy[0]
    yi = K.cos(xy[1])
    yq = -K.sin(xy[1])
    z = K.zeros_like(x)
    zi = K.expand_dims( x[:,:,0]*yi - x[:,:,1]*( yq)  )
    zq = K.expand_dims( x[:,:,0]*(yq) + x[:,:,1]*yi)
    z = K.concatenate([zi,zq])
    return z

def rotate_n(xy):
    x = xy[0]
    yi = K.cos(xy[1])
    yq = K.sin(xy[1])
    z = K.zeros_like(x)
    zi = K.expand_dims( x[:,:,0,:]*yi - x[:,:,1,:]*( yq)  )
    zq = K.expand_dims( x[:,:,0,:]*(yq) + x[:,:,1,:]*yi)
    z = K.concatenate([zi,zq])
    return z

def rotateCos(xCos):
    x = xCos[0]
    y = K.l2_normalize(xCos[1], axis=-1)
    yi = y[:,:,0]
    yq = y[:,:,1]
    z = K.zeros_like(x)
    zi = K.expand_dims( x[:,:,0]*yi - x[:,:,1]*( yq)  )
    zq = K.expand_dims( x[:,:,0]*(yq) + x[:,:,1]*yi)
    z = K.concatenate([zi,zq],axis = 2)
    return z


def merge_conv(x,F):
    # https://stackoverflow.com/questions/42068999/tensorflow-convolutions-with-different-filter-for-each-sample-in-the-mini-batch
    
    
    # F has shape (MB, fh, fw, channels, out_channels)
    # REM: with the notation in the question, we need: channels_img==channels
    

    MB = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    channels_img = tf.shape(x)[3]
    
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]
    channels = tf.shape(F)[3]
    out_channels  = tf.shape(F)[4]
    
#     (MB, H, W, channels_img) = tf.shape(x)
#     (MB, fh, fw, channels, out_channels) = tf.shape(F)
    
    F = tf.transpose(F, [1, 2, 0, 3, 4])
    F = tf.reshape(F, [fh, fw, channels*MB, out_channels])

    inp_r = tf.transpose(x, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
    inp_r = tf.reshape(inp_r, [1, H, W, MB*channels_img])

    padding = "SAME" #or "VALID"
    out = tf.nn.depthwise_conv2d(
              inp_r,
              filter=F,
              strides=[1, 1, 1, 1],
              padding=padding) # here no requirement about padding being 'VALID', use whatever you want. 
    # Now out shape is (1, H-fh+1, W-fw+1, MB*channels*out_channels), because we used "VALID"

    if padding == "SAME":
        out = tf.reshape(out, [H, W, MB, channels, out_channels])
    if padding == "VALID":
        out = tf.reshape(out, [H-fh+1, W-fw+1, MB, channels, out_channels])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
                           
    return out


def merge_conv_valid(x,F):
    # https://stackoverflow.com/questions/42068999/tensorflow-convolutions-with-different-filter-for-each-sample-in-the-mini-batch
    
    
    # F has shape (MB, fh, fw, channels, out_channels)
    # REM: with the notation in the question, we need: channels_img==channels
    

    MB = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    channels_img = tf.shape(x)[3]
    
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]
    channels = tf.shape(F)[3]
    out_channels  = tf.shape(F)[4]
    
#     (MB, H, W, channels_img) = tf.shape(x)
#     (MB, fh, fw, channels, out_channels) = tf.shape(F)
    
    F = tf.transpose(F, [1, 2, 0, 3, 4])
    F = tf.reshape(F, [fh, fw, channels*MB, out_channels])

    inp_r = tf.transpose(x, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
    inp_r = tf.reshape(inp_r, [1, H, W, MB*channels_img])

    padding = "VALID"
    out = tf.nn.depthwise_conv2d(
              inp_r,
              filter=F,
              strides=[1, 1, 1, 1],
              padding=padding) # here no requirement about padding being 'VALID', use whatever you want. 
    # Now out shape is (1, H-fh+1, W-fw+1, MB*channels*out_channels), because we used "VALID"

    if padding == "SAME":
        out = tf.reshape(out, [H, W, MB, channels, out_channels])
    if padding == "VALID":
        out = tf.reshape(out, [H-fh+1, W-fw+1, MB, channels, out_channels])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
                           
    return out

    

def noPhaseMSE(yTrue,xPred):
    pkt_size = 1024 
    
    yr = yTrue[:,:,0]
    yi = yTrue[:,:,1]
    xr = xPred[:,:,0]
    xi = xPred[:,:,1]
    
    xhx = K.mean(xr**2+xi**2,axis = -1)
    yhy = K.mean(yr**2+yi**2,axis = -1)
    xhyr = K.mean(xr*yr - (-xi)*  yi,axis = -1)
    xhyi = K.mean(xr*yi + (-xi)*yr,axis = -1)
    loss = (xhx  + yhy -   2*K.sqrt(xhyr**2 + xhyi**2 + K.epsilon()))*128/pkt_size
    return loss

def cmae(yTrue,xPred):
    pkt_size = 1024 
    
    yr = yTrue[:,:,0]
    yi = yTrue[:,:,1]
    xr = xPred[:,:,0]
    xi = xPred[:,:,1]

    xhyr = xr-yr
    xhyi = xi-yi
    loss = K.abs(K.sqrt(xhyr**2 + xhyi**2 + K.epsilon()))*128/pkt_size
    return loss





def conv_regularizer2(act):
    conv_filter_len = 65
    # print(act.shape)
    diff = act[:,1:] - act[:,0:-1]
    sym = act[:,0:conv_filter_len//2] - act[:,conv_filter_len+1:conv_filter_len//2:-1]
    diff_reg =  K.sum(diff**2)
    sym_reg = K.sum(K.abs(sym))
#     center_reg = K.sum(act[:,0:80]**2) + K.sum(act[:,256:256-80-1]**2) 
    l2_reg =  K.sum(act**2)
    l1_reg =  K.sum(K.abs(act))
    non_zero = K.sum(K.maximum(1-K.sum(act**2,axis=-1),0))
    return 1e-6*diff_reg + 1e-6 * sym_reg +  0*l1_reg + 0*l2_reg + 0*non_zero