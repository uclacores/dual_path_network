#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')



get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU


# In[3]:


from frm_dataset_loader import load_dataset
from conf_dataset_1 import *


# In[4]:


fname =  'datasets/valid_1.dat'

(comb_valid,carrier_valid,clean_valid,fading_valid,_,
         freq_valid,timing_offNum_valid,timing_step_valid,
         _,mod_valid,snr_valid) = load_dataset(fname,max_sps,len(mod_list))


# In[5]:


fname = 'datasets/test_1.dat'

(comb_test,carrier_test,clean_test,fading_test,raw_test,
         freq_test,timing_offNum_test,timing_step_test,
         coeff_test,mod_test,snr_test) = load_dataset(fname,max_sps,len(mod_list))


# In[6]:


from frm_nn_zoo_01 import create_dualPath

pkt_size_net = pkt_size
nn_all = create_dualPath(pkt_size = pkt_size_net,n_mods=len(mod_list))
nn_all.summary()


# In[7]:


from frm_train_generator import train_generator




gen = train_generator(100,64,pkt_size,max_sps,mod_list,
                   sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng)
len(gen.__getitem__(0)[1])


# In[8]:


# nn_all.load_weights('../models/033.h5')

import keras

patience = 10
n_epochs = 100
decay_rate = None #0.8
decay_step = None #25

def step_decay(epoch,lr):
    decay_rate = decay_rate
    decay_step = decay_step
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr



filepath = 'tmp/tmp_'+GPU

import time

c=[keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0, patience=patience, verbose=1, mode='auto'),
 keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True),]
  #keras.callbacks.LearningRateScheduler(step_decay)]


indx_valid = slice(None,None)
batch_size = 200



samples_per_epoch = 800000
gen = train_generator(samples_per_epoch,batch_size,pkt_size,max_sps,mod_list,
                   sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng)

history = nn_all.fit_generator(gen,
                            epochs = n_epochs,callbacks=c,
                            workers=10, use_multiprocessing=True,
                            validation_data = 
                            (comb_valid[indx_valid],[freq_valid[indx_valid],fading_valid[indx_valid],fading_valid[indx_valid],
                                                    clean_valid[indx_valid],timing_step_valid[indx_valid],
                                                    timing_offNum_valid[indx_valid],mod_valid[indx_valid]]))


# In[9]:


plt.plot(history.history['loss'][1:])
plt.plot(history.history['val_loss'][1:])
plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
print(history.history)


# In[10]:


filepath = 'tmp/tmp_'+GPU
nn_all.load_weights(filepath)


# In[11]:


op_eval = nn_all.evaluate(comb_test,[freq_test,fading_test,fading_test,clean_test,timing_step_test,timing_offNum_test,mod_test])
print(op_eval)


# In[12]:


import datetime

outputs = {}
outputs['dataset_params'] = [pkt_size,max_sps,mod_list,sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng]
outputs['date'] =  f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S%z}'
outputs['history'] = history.history
outputs['train_params'] = [patience,n_epochs,decay_rate,decay_step]
outputs['op_eval'] = op_eval


# In[ ]:





# In[14]:


FNAME = '001'


# In[15]:


import pickle
with open(f'outputs/{FNAME}.pkl','wb') as f:
    pickle.dump(outputs,f)


# In[16]:


nn_all.save_weights(f'models/{FNAME}.h5')

# nn_demod.save('models/{Di}_sig_data.h5'.format(",".join(mod_list)))


# In[ ]:




