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


from frm_dataset_loader import load_dataset
from conf_dataset_1 import *


# In[3]:


fname = 'datasets/test_1.dat'

(_,_,_,_,_,
         freq_test,timing_offNum_test,timing_step_test,
         _,mod_test,snr_test) = load_dataset(fname,max_sps,len(mod_list))


# In[4]:


with open(f'outputs/003.pkl','rb') as f:
    res_dsp = pickle.load(f)


# In[5]:


with open(f'outputs/005.pkl','rb') as f:
    res_dpn = pickle.load(f)


# In[6]:


test_ser_genie = res_dsp['genie'] 


# In[7]:


(pred_freq,pred_timing_step,pred_timing_off,pred_mod_dec, nmse_all,nmse2_all,nmse2t_all,test_ser) = res_dpn['dpn'] 


# In[8]:


from frm_eval_utils import ecdf,filter_mod_snr


# In[10]:


notnan = lambda x : x[np.logical_not(np.isnan(x))]
srt_genie,prob_genie = ecdf(notnan(test_ser_genie))
srt_dpn,prob_dpn = ecdf(notnan(test_ser))


# In[11]:


plt.figure(dpi=150)
plt.plot(srt_dpn,prob_dpn)
plt.plot(srt_genie,prob_genie)
plt.xlabel('SER')
plt.ylabel('P(SER<abscissa)')
plt.legend(['DPN','DSP','Genie'])


# In[12]:


clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']


# In[13]:


from frm_modulations import linear_mod_list
def calc_ecdf(mod_subs):
    res_dpn = []
    res_genie = []

    plt.figure(dpi = 150)
    for i,mod in enumerate(mod_subs):
        if mod in linear_mod_list:
            srt,prob = ecdf(filter_mod_snr(test_ser,snr_val,mod_list.index(mod),np.argmax(mod_test,-1),snr_test))
            res_dpn.append((srt,prob))
            srt,prob = ecdf(filter_mod_snr(test_ser_genie,snr_val,mod_list.index(mod),np.argmax(mod_test,-1),snr_test))
            res_genie.append((srt,prob))
    return (res_dpn,res_genie)


# In[16]:



def plot_ecdf(ecdfs,mod_subs,snr_val):
    (res_dpn,res_genie) = ecdfs
    plt.figure(dpi = 150)
    for i,mod in enumerate(mod_list_subs):
        if mod in linear_mod_list:
            srt,prob = res_dpn[i]
            plt.plot(srt,prob,color = clrs[i],label = mod)
            srt,prob = res_genie[i]
            plt.plot(srt,prob,color = clrs[i],linestyle = '--')
    plt.legend()
    plt.xlabel('SER')
    plt.ylabel('P(SER<abscissa)')
    plt.title(f'SNR {snr_val*2}')


# In[17]:


snr_val = snr_list[2]
mod_list_subs = mod_list[0:3]
ecdfs  = calc_ecdf(mod_list_subs)
plot_ecdf(ecdfs,mod_list_subs,snr_val)


mod_list_subs = mod_list[0:3]
ecdfs  = calc_ecdf(mod_list_subs)
plot_ecdf(ecdfs,mod_list_subs,snr_val)

mod_list_subs = mod_list[3:8]
ecdfs  = calc_ecdf(mod_list_subs)
plot_ecdf(ecdfs,mod_list_subs,snr_val)

mod_list_subs = mod_list[8:11]
ecdfs  = calc_ecdf(mod_list_subs)
plot_ecdf(ecdfs,mod_list_subs,snr_val)

mod_list_subs = mod_list[11:14]
ecdfs  = calc_ecdf(mod_list_subs)
plot_ecdf(ecdfs,mod_list_subs,snr_val)


# In[18]:


mod_list_subs = mod_list[3:6]
ecdfs_sub = calc_ecdf(mod_list_subs)
plot_ecdf(ecdfs_sub,mod_list_subs,snr_val)



plt.plot([],[],'k-',label='DPN')
plt.plot([],[],'k--',label='Genie')
plt.legend(ncol = 2)


# In[ ]:





# In[22]:


def calc_mod_snr(mod_subs,snr_list):
    res_dpn = np.zeros((len(mod_subs),len(snr_list)))
    res_genie = np.zeros((len(mod_subs),len(snr_list)))
    for j,mod in enumerate(mod_subs):
        for i,snr_val in enumerate(snr_list):
                res_dpn[j,i] = np.mean(filter_mod_snr(test_ser,snr_val,mod_list.index(mod),np.argmax(mod_test,-1),snr_test)>0)
                res_genie[j,i] = np.mean(filter_mod_snr(test_ser_genie,snr_val,mod_list.index(mod),np.argmax(mod_test,-1),snr_test)>0)
    return res_dpn,res_genie


# In[24]:


res_dpn = []
res_genie = []
res_blind = []

mod_subs = ['bpsk','qpsk']


res_dpn,res_genie = calc_mod_snr(mod_subs,snr_list)

plt.figure(dpi = 150)
plt.plot(snr_list*2,res_dpn.T)
plt.gca().set_prop_cycle(None)
plt.plot(snr_list*2,res_genie.T,'-.')

legend1 = plt.legend(mod_subs)
plt.ylim([0,1])
plt.xlabel('SNR')
plt.ylabel('Packet Error Rate')



plt.plot([],[],'k-',label='DPN')
plt.plot([],[],'k--',label='Genie')
plt.legend(loc='upper center')
plt.gca().add_artist(legend1)


# In[ ]:




