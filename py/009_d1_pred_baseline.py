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


fname = 'datasets/test_1.dat'

(comb_test,carrier_test,clean_test,fading_test,raw_test,
         freq_test,timing_offNum_test,timing_step_test,
         coeff_test,mod_test,snr_test) = load_dataset(fname,max_sps,len(mod_list))


# In[6]:


mod_test_dec = np.argmax(mod_test,-1)


# In[7]:


from frm_train_generator import train_generator_mod as train_generator


# In[8]:


from frm_nn_baseline import *


# In[9]:


gru = create_model_gru(len(mod_list),pkt_size)


# In[10]:


filepath = f'models/004_gru.h5'

gru.load_weights(filepath)


# In[11]:


pred = gru.predict(comb_test)
gru_mod_dec = np.argmax(pred,-1)

print(np.mean(gru_mod_dec == mod_test_dec))


# In[12]:


import datetime

outputs = {}
outputs['dataset_params'] = [pkt_size,max_sps,mod_list,sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng]
outputs['date'] =  f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S%z}'
outputs['pred'] = [gru_mod_dec]


# In[13]:


FNAME = '009'


# In[14]:


import pickle
with open(f'outputs/{FNAME}.pkl','wb') as f:
    pickle.dump(outputs,f)


# In[ ]:





# In[ ]:





# In[ ]:




