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
GPU = '0'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= GPU


# In[3]:


import keras


# In[4]:


from frm_dataset_loader import load_dataset
from conf_dataset_1 import *


# In[5]:


fname =  'datasets/valid_1.dat'

(comb_valid,carrier_valid,clean_valid,fading_valid,_,
         freq_valid,timing_offNum_valid,timing_step_valid,
         _,mod_valid,snr_valid) = load_dataset(fname,max_sps,len(mod_list))


# In[6]:


fname = 'datasets/test_1.dat'

(comb_test,carrier_test,clean_test,fading_test,raw_test,
         freq_test,timing_offNum_test,timing_step_test,
         coeff_test,mod_test,snr_test) = load_dataset(fname,max_sps,len(mod_list))


# In[7]:


from frm_train_generator import train_generator_mod as train_generator


# In[8]:


from frm_nn_baseline import *


# In[9]:


FNAME = '004'


# In[10]:


indx_valid = slice(None)

batch_size = 128

samples_per_epoch = 800000

gen = train_generator(samples_per_epoch,batch_size,pkt_size,max_sps,mod_list,
                   sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng)
gen.__getitem__(0)
patience = 10
n_epochs = 400
nworkers = 10


# In[ ]:


gru = create_model_gru(len(mod_list),pkt_size)


# In[ ]:


filepath = f'models/{FNAME}_gru.h5'
c=[ keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True),
  keras.callbacks.EarlyStopping(monitor='val_loss',  patience=patience)]


gen.first_run = True

gru_history = gru.fit_generator(gen,
                            epochs = n_epochs,callbacks=c,
                            workers=nworkers, use_multiprocessing=True,
                            validation_data = 
                            (comb_valid[indx_valid], mod_valid[indx_valid]))


# In[ ]:


import datetime

outputs = {}
outputs['dataset_params'] = [pkt_size,max_sps,mod_list,sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng]
outputs['date'] =  f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S%z}'
outputs['history'] = [gru_history.history]
outputs['train_params'] = [patience,n_epochs]


# In[ ]:


import pickle
with open(f'outputs/{FNAME}.pkl','wb') as f:
    pickle.dump(outputs,f)


# In[12]:


import pickle
with open(f'outputs/{FNAME}.pkl','rb') as f:
    outputs = pickle.load(f)
[gru_history.history]=outputs['history']


# In[ ]:




