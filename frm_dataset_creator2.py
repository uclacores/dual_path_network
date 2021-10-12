#from frm_generate_data_np import *
import numpy as np
from numpy import pi,sqrt

model_folder="models/freq_2019_07_02_2/"

from frm_modulations import mod_list,cont_phase_mod_list,linear_mod_const
# import tensorflow as tf
import datetime
import pickle
import sys
import copy
from frm_dataset_creator import *

from numba import jit

from frm_modulations_fast import modulate_symbols_fast,modulate_symbols
# In[199]:



def func(my_dict):
    # print(my_dict)
    return generate_dataset_sig2(**my_dict)

def generate_dataset_sig2_parallel(n_samples, pkt_size,max_sps,mod_list,sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng, complex_fading = False,freq_in_hz = False,
    seed = None, fname = None, version = 1,nthreads = 10 ): #1e4    
        args_in = locals()
        args_in.pop('nthreads',None)

        rand_step =374861
        args_list = []
        for i in range(nthreads):
            args_list.append(copy.deepcopy(args_in))


        if args_in['seed'] is not None:
            for indx,args in enumerate(args_list):
                args['seed'] = args_in['seed'] + indx * rand_step

        get_tmp_name = lambda base, indx : "{}_{}".format(base, indx) 

        # if fname is not None:
        #     base_name = fname
        # else:
        #     base_name = 'tmp/dataset'

        # base_name = '/tmp/dataset{}'.format(np.random.randint(0,1000000))

        for indx,args in enumerate(args_list):
            args['fname'] = None #get_tmp_name(base_name,indx)
            args['n_samples'] = args_in['n_samples']//nthreads
        p = Pool(nthreads)
        datasets = p.map(func, args_list)

        # with open(get_tmp_name(base_name,0),'rb') as f:
        #     dataset = pickle.load(f)

        dataset_out = datasets[0]
        for i in range(1,nthreads):
            dataset_i = datasets[i]

            for k1 in dataset_out.keys():
                if isinstance(dataset_out[k1],dict):
                    for k2 in dataset_out[k1].keys():
                        # print(k1,k2)
                        if k1!='args' and k1!='time':
                            dataset_out[k1][k2] = np.append(dataset_out[k1][k2],dataset_i[k1][k2],axis = 0) 


        dataset_out['args'] = args_in
        dataset_out['time'] = str(datetime.datetime.now())
        if fname is not None:
            with open(fname,'wb') as f:
                pickle.dump(dataset_out,f)

        return dataset_out


def generate_dataset_sig2(n_samples, pkt_size,max_sps,mod_list,sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng,complex_fading = False, freq_in_hz = False,
    seed = None, fname = None, version = 1):
    
    args = locals()

    if seed is not None:
        np.random.seed(seed)
    comb_v = np.zeros((n_samples,pkt_size,2))
    carrier_v = np.zeros((n_samples,pkt_size,2))
    fading_v = np.zeros((n_samples,pkt_size,2))
    clean_v = np.zeros((n_samples,pkt_size,2))
    timing_v = np.zeros((n_samples,pkt_size,2))
    raw_v =  np.zeros((n_samples,pkt_size,2))
    mod_v = np.zeros((n_samples,len(mod_list)))
    if not complex_fading:
        coeff = np.zeros((n_samples,6))
    else:
        coeff = np.zeros((n_samples,6),dtype='complex')

    mod_index = np.random.choice(len(mod_list),(n_samples,)).astype(np.int_)
    mod_v[range(n_samples),mod_index] = 1
    sps = np.random.uniform(sps_rng[0],sps_rng[1],(n_samples,))
    pulse_ebw = np.random.choice(pulse_ebw_list,(n_samples,))
    timing_offset = np.random.uniform(timing_offset_rng[0],timing_offset_rng[1],(n_samples,))
    fading_spread = np.random.uniform(fading_spread_rng[0],fading_spread_rng[1],(n_samples,))
    freq_err = np.random.uniform(freq_err_rng[0],freq_err_rng[1],(n_samples,))
    phase_err = np.random.uniform(phase_err_rng[0],phase_err_rng[1],(n_samples,))
    if np.array(snr_rng).size==2:
        snr = np.random.uniform(snr_rng[0],snr_rng[1],(n_samples,))
    else:
        snr = np.random.choice(snr_rng,(n_samples,)) 
     



    
    progress_step = 1000
    a = datetime.datetime.now()
    strt_time = copy.deepcopy(a)
    for samp_indx in range(n_samples):

        


        mod = mod_list[mod_index[samp_indx]]
        op = create_sample_fast( mod = mod,pkt_len = pkt_size,sps=sps[samp_indx],pulse_ebw = pulse_ebw[samp_indx],
              timing_offset = timing_offset[samp_indx], 
              fading_spread = fading_spread[samp_indx], 
              freq_err = freq_err[samp_indx], phase_err =phase_err[samp_indx],    
              snr = snr[samp_indx], max_sps = max_sps, complex_fading = complex_fading, freq_in_hz = freq_in_hz,
              seed = None)
        mod_v[:,0] = 1
        comb_v[samp_indx] ,carrier_v[samp_indx],fading_v[samp_indx],clean_v[samp_indx],timing_v[samp_indx],raw_v[samp_indx],coeff[samp_indx] = op

        if samp_indx%progress_step == 0 and samp_indx>0:
                b = datetime.datetime.now()
                diff_time = b-a
     
                # the exact output you're looking for:
                sys.stdout.write("\rGenerated {} out of {} ({:.1f}%), Elapsed {} , estimated {}".format(samp_indx,n_samples, float(samp_indx)/n_samples*100, b-strt_time , (n_samples-samp_indx)*diff_time /progress_step ))
                sys.stdout.flush()
                a = copy.deepcopy(b)
            
    op ={'sig':{},'params':{},'data':{}}

    op['sig']['comb'] =  comb_v


    op['sig']['timing_fading_carrier'] = carrier_v
    op['sig']['timing_fading'] = fading_v
    op['sig']['timing'] = clean_v

    op['params']['mod'] = mod_index


    op['params']['fading_spread'] = fading_spread
    op['params']['fading_taps'] = coeff


    op['params']['freq_off'] = freq_err
    op['params']['phase_off'] = phase_err 



    op['params']['timing_off'] = timing_offset
    op['params']['symb_rate'] = sps
    op['data']['binary_marking'] = timing_v
        

    op['params']['sps'] = sps
    op['params']['pulse_ebw'] = pulse_ebw


    op['sig']['timing_raw_unique'] = raw_v

    op['params']['snr'] = snr

    op['args'] = args
    op['time'] = str(datetime.datetime.now())
    op['version'] = version
    if fname is not None:
        with open(fname,'wb') as f:
            pickle.dump(op,f)


    return op




def create_sample( mod = 'bpsk',pkt_len = 128,sps=8,pulse_ebw = 0.35,
                  timing_offset = 0.5,
                  fading_spread = 1, 
                  freq_err = 0.0001, phase_err = np.pi,    
                  snr = 10, max_sps = 128, complex_fading = False, freq_in_hz = False,
                  seed = None):
    samp_rate = 1
    if seed is not None:
        np.random.seed(seed)
        
    if mod in cont_phase_mod_list:
        order = 2
    else: # Linear modulation
        order =  linear_mod_const[mod].size  
     
        
    n_symbols = int( (pkt_len)/(sps*0.5)) + 2

        
    data_symbs=np.random.randint(0,order,n_symbols)

        
    mag = timing_offset
    timing_offset = calc_timing_offset(mag, max_sps)

    timing_step =  int(max_sps/sps) 


    mod_symbs_max_sps=modulate_symbols(data_symbs,mod,max_sps,ebw = pulse_ebw)
    data_symbs_max_sps= np.repeat(data_symbs,max_sps)
        
        
    t_max_sps= np.arange(0,1.0*max_sps*n_symbols/samp_rate,1.0/samp_rate)
        
        

    transition_data_ideal = np.array(([1,]*max_sps + [0,]*max_sps) * int(n_symbols/2+1))



    

    mod_symbs_timing_err = simulate_timing_error(mod_symbs_max_sps,timing_offset,timing_step, pkt_len)

    data_symbs_timing_err = simulate_timing_error(data_symbs_max_sps,timing_offset,timing_step, pkt_len)
    mod_raw_symbs_timing_err = modulate_symbols(data_symbs_timing_err,mod,sps = 1, ebw = None, pulse_shape = None)

    t_timing_err = simulate_timing_error(t_max_sps,timing_offset,timing_step, pkt_len)

   

    marking_b_timing = simulate_timing_error(transition_data_ideal,timing_offset,timing_step, pkt_len)
    
    transition_data_timing = simulate_timing_error(transition_data_ideal,timing_offset,timing_step, pkt_len+1)
    transition_data_timing = np.abs(np.diff(transition_data_timing)).astype('int')
    
    mod_raw_unique_symbs_timing_err = mod_raw_symbs_timing_err*transition_data_timing
    mod_raw_unique_symbs_timing_err[transition_data_timing==0]=np.nan+1j*np.nan


    if not complex_fading:
        coeff = generate_fading_taps(max_sps / timing_step, fading_spread)
        mod_symbs_timing_fading = simulate_fading_channel(mod_symbs_timing_err, coeff)
    else:
        coeff=generate_complex_fading_taps(max_sps / timing_step, fading_spread)
        mod_symbs_timing_fading = simulate_fading_channel_complex(mod_symbs_timing_err, coeff)
               
    if not freq_in_hz:
        t_freq = t_timing_err
    else:
        t_freq = np.arange(t_timing_err.size)
        
    mod_symbs_timing_fading_freq_err = simulate_frequency_error(mod_symbs_timing_fading,t_freq,freq_err,phase_err)
    carrier_timing_err = simulate_frequency_error(1.0,t_freq,freq_err,phase_err)
  
    mod_symbs_timing_fading_freq_noise = add_noise(mod_symbs_timing_fading_freq_err,snr)
   
    op = mod_symbs_timing_fading_freq_noise
    


    comb = assign_iq2(mod_symbs_timing_fading_freq_noise)
    carrier = assign_iq2(mod_symbs_timing_fading_freq_err)
    fading = assign_iq2(mod_symbs_timing_fading)
    clean = assign_iq2(mod_symbs_timing_err)#assign_iq2(mod_symbs_max_sps)#


    timing =  np.zeros((pkt_len,2))
    timing[range(pkt_len),marking_b_timing] = 1
    
    raw = assign_iq2(mod_raw_unique_symbs_timing_err)
    return   (comb ,carrier,fading,clean,timing,raw,coeff)


@jit(nopython=True)
def create_marking(max_sps,timing_step,timing_offset,pkt_len):
    x = np.zeros(pkt_len+1,dtype=np.int_)
    timing_offset = int(timing_offset)
    indx = int(timing_offset)
    state = True
    prev_max_sps = indx% max_sps
    for i in range(0,x.size):
        x[i] = state
        indx = indx +timing_step
        cur_max_sps = indx%max_sps
        if cur_max_sps<prev_max_sps:
            state = not state
        prev_max_sps = cur_max_sps
    return x

def create_sample_fast( mod = 'bpsk',pkt_len = 128,sps=8,pulse_ebw = 0.35,
                  timing_offset = 0.5,
                  fading_spread = 1, 
                  freq_err = 0.0001, phase_err = np.pi,    
                  snr = 10, max_sps = 128,complex_fading = False, freq_in_hz = False,
                  seed = None):
    samp_rate = 1
    if seed is not None:
        np.random.seed(seed)
        
    if mod in cont_phase_mod_list:
        order = 2
    else: # Linear modulation
        order =  linear_mod_const[mod].size  
     
        
    n_symbols = int( (pkt_len)/(sps*0.5)) + 2

        
    data_symbs=np.random.randint(0,order,n_symbols)

        
    
    mag = timing_offset
    timing_offset = calc_timing_offset(mag, max_sps)

    timing_step =  int(max_sps/sps)

    mod_symbs_max_sps=modulate_symbols_fast(data_symbs,mod,max_sps,timing_offset,timing_step,ebw = pulse_ebw)

    data_symbs_max_sps= np.repeat(data_symbs,max_sps)

        
        
    t_max_sps= np.arange(0,1.0*max_sps*n_symbols/samp_rate,1.0/samp_rate)
        
        





    mod_symbs_timing_err = mod_symbs_max_sps[:pkt_len]

    data_symbs_timing_err = simulate_timing_error(data_symbs_max_sps,timing_offset,timing_step, pkt_len)

    mod_raw_symbs_timing_err = modulate_symbols(data_symbs_timing_err,mod,sps = 1, ebw = None, pulse_shape = None)


    t_timing_err = simulate_timing_error(t_max_sps,timing_offset,timing_step, pkt_len)


    transition_data_timing = create_marking(max_sps,timing_step,timing_offset,pkt_len)


    marking_b_timing = transition_data_timing[:-1]
    
    transition_data_timing = np.abs(np.diff(transition_data_timing)).astype('int')

    
    mod_raw_unique_symbs_timing_err = mod_raw_symbs_timing_err*transition_data_timing
    mod_raw_unique_symbs_timing_err[transition_data_timing==0]=np.nan+1j*np.nan


    if not complex_fading:
        coeff = generate_fading_taps(max_sps / timing_step, fading_spread)
        mod_symbs_timing_fading = simulate_fading_channel(mod_symbs_timing_err, coeff)
    else:
        coeff=generate_complex_fading_taps(max_sps / timing_step, fading_spread)
        mod_symbs_timing_fading = simulate_fading_channel_complex(mod_symbs_timing_err, coeff)
    
    
    if not freq_in_hz:
        t_freq = t_timing_err
    else:
        t_freq = np.arange(t_timing_err.size)
               
    mod_symbs_timing_fading_freq_err = simulate_frequency_error(mod_symbs_timing_fading,t_freq,freq_err,phase_err)
    carrier_timing_err = simulate_frequency_error(1.0,t_freq,freq_err,phase_err)
  
    mod_symbs_timing_fading_freq_noise = add_noise(mod_symbs_timing_fading_freq_err,snr)
   
    op = mod_symbs_timing_fading_freq_noise
    


    comb = assign_iq2(mod_symbs_timing_fading_freq_noise)
    carrier = assign_iq2(mod_symbs_timing_fading_freq_err)
    fading = assign_iq2(mod_symbs_timing_fading)
    clean = assign_iq2(mod_symbs_timing_err)


    timing =  np.zeros((pkt_len,2))
    timing[range(pkt_len),marking_b_timing.astype(np.int_)] = 1
    
    raw = assign_iq2(mod_raw_unique_symbs_timing_err)
    return   (comb ,carrier,fading,clean,timing,raw,coeff)


def assign_iq2( complex_vec):
    op_vec = np.zeros((complex_vec.shape[0],2))
    op_vec[:,0] = np.real(complex_vec)
    op_vec[:,1] = np.imag(complex_vec)
    return op_vec






if __name__ == '__main__':
    test_data_sig_parallel()