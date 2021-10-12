import tensorflow as tf
import keras
import keras as keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import regularizers
import keras.backend as K

from frm_nn_functions import *


ap = lambda x,y: x+'_'+y
# Decreases parameters extractors. Decreased process. Added resnets in get_mod
def resnet(x,w,f,name):
    nm = lambda x : ap(name,x)
    x = Conv2D(w,(1,1),activation=None,padding = 'same',name = nm('conv1') )(x)
    x_b = x
    x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv2'))(x)
    x = BatchNormalization(name = nm('bn'))(x)
    x = Activation(activation='relu',name = nm('act1'))(x)
    x = Conv2D(w,f,activation=None,padding = 'same',name = nm('conv3'))(x)
    x = BatchNormalization(name = nm('bn2'))(x)
    x = Add(name = nm('add'))([x,x_b])
    x = Activation(activation='relu',name = nm('act2'))(x)
    return x


denoise_filter_len = 32
fading_filter_len = 32
drop_rate = 0.15

def extract_features(x,pkt_size,parent_name):
    name = ap(parent_name,'extrct')
    nm = lambda x : ap(name,x)
    x = Reshape((pkt_size,2,1),name = nm('rshp'))(x)
    x = resnet(x,16,(3,1),nm('rsnt1'))
    x = MaxPooling2D((2,1),name=nm('mp1'))(x)
    x = resnet(x,32,(3,2),nm('rsnt2'))
    x = MaxPooling2D((1,2),name=nm('mp2'))(x)
    return x

def process_features(x,parent_name):
    name = ap(parent_name,'prcss')
    nm = lambda x : ap(name,x)
    x = resnet(x,32,(3,1),nm('rsnt1'))
#     x = resnet(x,32,(3,1),nm('rsnt2'))
    return x

def get_denoise_filter(x):
    nm = lambda x : ap('gtDenoise',x)
    x = resnet(x,16,(3,1),nm('rsnt1'))
    x = MaxPooling2D((2,1),name = nm('mp1'))(x)
    x = resnet(x,8,(3,1),nm('rsnt2'))
    x = MaxPooling2D((2,1),name = nm('mp2'))(x)
    x = Conv2D(denoise_filter_len,(3,1),activation=None,padding='same',name = nm('conv'))(x)
#     x = Dropout(drop_rate,name=nm('drp'))(x)
    x = GlobalAveragePooling2D(name='gp_denoise')(x)
    return x  



def denoise(x,x_filter,pkt_size):
    nm = lambda x : ap('appDenoise',x)
    
    x = Reshape((pkt_size,2,1),name = nm('rshp1'))(x)
    x_filter = Reshape((denoise_filter_len,1,1,1),name = nm('rshp2'))(x_filter)
    x = Lambda(lambda x: merge_conv(x[0],x[1]),name = nm('mergeConv'))([x,x_filter])
    x = Reshape((pkt_size,2),name = nm('rshp3'))(x)
    return x

def get_freq(x):
    nm = lambda x : ap('gtFreq',x)
    x = resnet(x,32,(3,1),nm('rsnt1'))
    x = MaxPool2D((2,1),name=nm('mp1'))(x)
    x = resnet(x,32,(3,1),nm('rsnt2'))
    x = MaxPool2D((2,1),name=nm('mp2'))(x)
    x = resnet(x,16,(3,1),nm('rsnt3'))
    x = Conv2D(1,(1,1),activation=None,name=nm('conv'))(x)
    x = GlobalAveragePooling2D()(x)
#     x = Reshape((pkt_size//2,32))(x)
#     x = CuDNNGRU(64,return_sequences=False,name=nm('GRU1'))(x)
#     x = Dense(1,activation=None,activity_regularizer=keras.regularizers.l2(1e-7))(x)
    return x  

def corr_freq(x,x_freq,pkt_size):
    nm = lambda x : ap('apFreq',x)
    if pkt_size == -1:
        x_phase = Lambda(lambda xf: xf * K.arange(K.shape(x)[1],dtype='float32') ,name='phase')(x_freq)
    else:
        x_phase = Lambda(lambda xf: xf * K.arange(pkt_size,dtype='float32') ,name='phase')(x_freq)
    x_phase = Reshape((pkt_size,),name = nm('rshp1'))(x_phase)
    x = Lambda(rotate,name = nm('rotate')) ([x,x_phase])
    x = Reshape((pkt_size,2),name = nm('rshp2'))(x)
    return x




def get_timing_sps(x,pkt_size,depth=32):
    nm = lambda x : ap('timeSps',x)
    x = Reshape((pkt_size//2,1,depth),name = nm('rshp1'))(x)
    x = MaxPool2D((2,1),name=nm('mp1'))(x)
    x = Conv2D(32,(3,2),activation='relu',padding='same',name = nm('conv1'))(x)
    x = Conv2D(32,(3,2),activation='relu',padding='same',name = nm('conv2'))(x)
    x = MaxPool2D((2,1),name=nm('mp2'))(x)
    x = Conv2D(32,(3,1),activation='relu',padding='same',name = nm('conv3'))(x)
    x = Conv2D(32,(3,1),activation='relu',padding='same',name = nm('conv4'))(x)
    x = Conv2D(1,(1,1),activation=None,padding='same',name = nm('conv5'))(x)
    x = GlobalAveragePooling2D(name='gp_time_sps')(x)
    return x

def get_timing_off(x,pkt_size,depth=32):
    nm = lambda x : ap('timeOff',x)
    x = Reshape((pkt_size//2,1,depth),name = nm('rshp1'))(x)
    x = MaxPool2D((2,1),name=nm('mp1'))(x)
    x = Conv2D(32,(3,2),activation='relu',padding='same',name = nm('conv1'))(x)
    x = Conv2D(32,(3,2),activation='relu',padding='same',name = nm('conv2'))(x)
    x = MaxPool2D((2,1),name=nm('mp2'))(x)
    x = Conv2D(32,(3,1),activation='relu',padding='same',name = nm('conv3'))(x)
    x = Conv2D(32,(3,1),activation='relu',padding='same',name = nm('conv4'))(x)
    x = Conv2D(1,(1,1),activation=None,padding='same',name = nm('conv5'))(x)
    x = GlobalAveragePooling2D(name='gp_time_off')(x)
    return x


def get_raw_params(x):
    nm = lambda x : ap('gtRaw',x)
    
    raw_filter_len = 32
    
    x = resnet(x,16,(3,1),nm('rsnt1'))
    x = MaxPooling2D((2,1),name =nm('mp1'))(x)
    xb = x
    x = resnet(x,8,(3,1),nm('rsnt2'))
    x = Conv2D(32,(1,1),activation=None,padding='same',name = nm('conv1'))(x)
    x = GlobalAveragePooling2D(name='gp_conv_filter')(x)
    x_filt = Dense(raw_filter_len,activation = 'relu',name = nm('dense1'))(x)
    x_filt = Dense(raw_filter_len,activation = None,activity_regularizer=keras.regularizers.l1(1e-7),
                  name = nm('dense27'))(x)
    return x_filt


def get_fading_filter(x):
    nm = lambda x : ap('gtFading',x)
    
    x = resnet(x,32,(3,1),nm('rsnt1'))
    x = MaxPooling2D((2,1),name =nm('mp1'))(x)
    x = resnet(x,16,(3,1),nm('rsnt2'))
    
    xr = Conv2D(fading_filter_len,(3,1),activation=None,padding='same',
               activity_regularizer = keras.regularizers.l1(1e-8),name = nm('conv_r'))(x)
    xr = GlobalAveragePooling2D( name = 'gp_fading_filter_r')(xr)
    
    xi = Conv2D(fading_filter_len,(3,1),activation=None,padding='same',
               activity_regularizer = keras.regularizers.l1(1e-8),name = nm('conv_i'))(x)
    xi = GlobalAveragePooling2D( name = 'gp_fading_filter_i')(xi)
    
    return (xr,xi) 


def conv_complex_mult(xr,xi):
    zr = K.expand_dims( xr[:,:,0] - xi[:,:,1] )
    zi = K.expand_dims( xi[:,:,0] + xr[:,:,1])
    z = K.concatenate([zr,zi])
    return z

def corr_fading(x,x_filter_r,x_filter_i,pkt_size):
    nm = lambda x : ap('apFading',x)
    
    x = Reshape((pkt_size,2,1),name=nm('rshp1'))(x)
    x = Lambda(lambda x: K.spatial_2d_padding( x, ((fading_filter_len-1,0), (0,0) )),name=nm('padr'))(x)
    
    
    x_filter_r = Reshape((fading_filter_len,1,1,1),name=nm('rshp2_r'))(x_filter_r)
    xr = Lambda(lambda x: merge_conv_valid(x[0],x[1]),name=nm('mergeConv_r'))([x,x_filter_r])
    print(xr.shape)
    xr = Reshape((pkt_size,2),name=nm('rshp3_r'))(xr)
    
    x_filter_i = Reshape((fading_filter_len,1,1,1),name=nm('rshp2_i'))(x_filter_i)
    xi = Lambda(lambda x: merge_conv_valid(x[0],x[1]),name=nm('mergeConv_i'))([x,x_filter_i])
    xi = Reshape((pkt_size,2),name=nm('rshp3_i'))(xi)
    
    z = Lambda(lambda x: conv_complex_mult(x[0],x[1]),name = nm('ComplxMult'))([xr,xi])
    
    print(z.shape)
    return z


def raw(x,x_conv,pkt_size):
    nm = lambda x : ap('apRaw',x)

    raw_filter_len = 65
    
    x_conv = Reshape((raw_filter_len,1,1,1),name = nm('rshp1'))(x_conv)
    x = Reshape((pkt_size,2,1),name = nm('rshp2'))(x)
    
    x = Lambda(lambda x: merge_conv(x[0],x[1]),name = nm('mergeConv'))([x,x_conv])
    x = Reshape((pkt_size,2),name = nm('rshp3'))(x)
    return x



def get_mod(x,n_mods):
    nm = lambda x : ap('mod',x)
#     print(x.shape)
    x = Reshape((-1,32))(x) # K.shape(x)[1] pkt_size//2
    x = CuDNNGRU(128,return_sequences=False,name=nm('GRU1'))(x)
    x = Dense(n_mods,activation = None,name =nm('Dense1'))(x)
    x = Activation('softmax',name = nm('sftmx'))(x)
    return x



def create_dualPath(pkt_size,n_mods):
    
    in_shp = [pkt_size,2]
    if pkt_size==-1:
        in_shp = [None,2]

    x_in = Input(shape=in_shp,name='sig_in')
    
    
    x_sig = x_in
    
    
    x_f_main = extract_features(x_in,pkt_size,'comb')

    x_f_main =  process_features(x_f_main,'comb')
    
    x_freq = get_freq(x_f_main)
    
    x_freq_out = Lambda(lambda x : x,name = 'freq')(x_freq)
    
    x_sig = corr_freq(x_sig,x_freq,pkt_size)
    x_mf_out = Lambda(lambda x : x,name = 'mf')(x_sig)
    x_sig = Lambda(lambda x: K.stop_gradient(x),name='freq_gs')(x_sig)
    
    x_f_freq = extract_features(x_sig,pkt_size,'freq')
    
    x_f_main = Concatenate()([x_f_main,x_f_freq])
    x_f_main =  process_features(x_f_main,'freq')
    
    
    x_denoise_filter = get_denoise_filter(x_f_main)
    x_sig = denoise(x_sig,x_denoise_filter,pkt_size)
    x_fading_out = Lambda(lambda x : x,name = 'fading')(x_sig)
    x_sig = Lambda(lambda x: K.stop_gradient(x),name='noise_gs')(x_sig)
    
    x_f_denoise = extract_features(x_sig,pkt_size,'noise')
    
    x_f_main = Concatenate()([x_f_main,x_f_denoise])
    x_f_main =  process_features(x_f_main,'noise')
    
    
    x_fading_r,x_fading_i = get_fading_filter(x_f_main)
    x_sig = corr_fading(x_sig,x_fading_r,x_fading_i,pkt_size)
    x_clean_out = Lambda(lambda x : x,name = 'clean')(x_sig)
    x_sig = Lambda(lambda x: K.stop_gradient(x),name='fading_gs')(x_sig)
    x_f_fading = extract_features(x_sig,pkt_size,'fading')
    
    x_f_main = Concatenate()([x_f_main,x_f_fading])
    x_f_main =  process_features(x_f_main,'fading')
    
    x_timing_sps = get_timing_sps(x_f_main,pkt_size)
    x_timing_sps_out = Lambda(lambda x : x,name = 'timing_sps')(x_timing_sps)
    
    x_timing_off = get_timing_off(x_f_main,pkt_size)
    x_timing_off_out = Lambda(lambda x : x,name = 'timing_off')(x_timing_off)

    

    x_f_main =  process_features(x_f_main,'mod')
    x_mod = get_mod(x_f_main,n_mods)
    x_mod = Lambda(lambda x : x,name = 'mod')(x_mod)

    model = Model(inputs = x_in, outputs =[x_freq_out,x_mf_out,x_fading_out,x_clean_out,x_timing_sps_out,
                                           x_timing_off_out,x_mod])
    model.compile(loss=['mae',noPhaseMSE,noPhaseMSE,noPhaseMSE,'mse','mse','categorical_crossentropy'], 
                  optimizer=keras.optimizers.Adam(0.001),
                    loss_weights=[500,0.001,1,5,2.0/4096,1.0/4096,1.0],
                  metrics={'freq':'mae','mf':noPhaseMSE,'fading':noPhaseMSE,'clean':noPhaseMSE,
                           'timing_sps':'mse','timing_off':'mse','mod':'categorical_accuracy'})    
    #model.summary()
    return model#,model_prep


