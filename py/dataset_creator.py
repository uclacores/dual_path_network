#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')


import sys

get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from frm_modulations import *
from frm_dataset_creator import *
from frm_dataset_creator2 import *
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150


from frm_dataset_loader import load_dataset
from conf_dataset_1 import *


# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
GPU = ""
os.environ["CUDA_VISIBLE_DEVICES"]=GPU


# In[3]:


fname =  'datasets/valid_1.dat'

dataset_valid = generate_dataset_sig2_parallel(10000, pkt_size,max_sps,mod_list,sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng,
    complex_fading=True, freq_in_hz = True,
    seed = 0, fname = fname, version = 1,nthreads = 40)



# In[4]:


fname = 'datasets/test_2.dat'

dataset_test = generate_dataset_sig2_parallel(100000, pkt_size,max_sps,mod_list,sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_list,
complex_fading=True, freq_in_hz = True,
seed = 1, fname = fname, version = 1,nthreads = 40)


# In[5]:


indx = 12
for k in dataset_test['sig'].keys():
    plt.figure()
    plt.plot(dataset_test['sig'][k][indx,:,0])
    plt.plot(dataset_test['sig'][k][indx,:,1])
    plt.title(k)
    plt.figure()


# In[ ]:




