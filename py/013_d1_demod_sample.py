#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')



get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import time


# In[2]:


import os
GPU = '0'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= GPU


# In[3]:


from frm_dataset_creator2 import create_sample_fast
from conf_dataset_1 import *


# In[5]:


from frm_nn_zoo_01 import create_dualPath

pkt_size_net = pkt_size
nn_all = create_dualPath(pkt_size = pkt_size_net,n_mods=len(mod_list))
# nn_all.summary()


# In[6]:


nn_all.load_weights('models/001.h5')


# In[7]:


ref_timing_offset = 0.3
ref_sps = 8
ref_freq = 0.01
ref_mod = 'qpsk'
ref_snr = 2.5
ref_fading = 0.3
max_sps = 64
seed = 3
ref_ebw = 0.35



x =  create_sample_fast( mod = ref_mod,pkt_len = pkt_size,sps=ref_sps,pulse_ebw = ref_ebw,
                  timing_offset = ref_timing_offset,
                  fading_spread = ref_fading, 
                  freq_err = ref_freq, phase_err = np.pi*0.34,    
                  snr = ref_snr, max_sps = max_sps, 
                        complex_fading=True, freq_in_hz = True, 
                  seed = seed)


ref_timing_step = int(max_sps/ref_sps)
ref_timing_offset_num = int(np.round(ref_timing_offset*max_sps))
    
(sig_comb ,sig_carrier,sig_fading,sig_clean,_,raw,coeff) = x


# In[ ]:





# In[8]:


y = nn_all.predict(sig_comb[None,:])
y_freq,y_noisy,y_fading,y_clean,y_timing_step,y_timing_off,y_mod  = y



y_timing_off = max(y_timing_off,0)
print('timing_off',ref_timing_offset_num,y_timing_off)
print('Timing Step',ref_timing_step,y_timing_step)
print('freq', (ref_freq),y_freq/(2*np.pi))

ip = y_clean[0]#x[0]
ref = raw


# In[9]:


cmp = lambda x : x[:,0] + 1j*x[:,1]
mx = lambda x : x/ np.sqrt(np.mean(np.sum(np.abs(x)**2,-1)))


# In[10]:



view_slc = slice(0,128)
plt.figure()
plt.plot(sig_comb[view_slc])
# plt.figure()
# plt.plot(sig_carrier[view_slc],'--')
# plt.plot(y_noisy[0,view_slc])

plt.figure()
plt.plot(mx(sig_fading[view_slc]))
# plt.plot(mx(y_fading[0,view_slc]))

plt.figure()
plt.plot(mx(sig_clean[view_slc]))
# plt.plot(mx(y_clean[0,view_slc]))


# In[11]:


plt.figure()
plt.ylim([-30,10])
plt.psd(cmp(mx(sig_carrier)))
plt.psd(cmp(mx(y_noisy[0])))


plt.figure()
plt.ylim([-30,10])
plt.psd(cmp(mx(sig_fading)))
plt.psd(cmp(mx(y_fading[0])))


plt.figure()
plt.ylim([-30,10])
plt.psd(cmp(mx(sig_clean)))
plt.psd(cmp(mx(y_clean[0])))
plt.ylim([-30,10])


# In[12]:


plt.figure()
plt.ylim([-30,15])
plt.psd(cmp(mx(sig_comb)),Fs=1)
plt.psd(cmp(mx(y_clean[0])),Fs=1)
plt.psd(cmp(mx(sig_clean)),Fs=1);
plt.legend(['Input','DPN','Ref'])

