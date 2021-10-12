#from frm_generate_data_np import *
import numpy as np
from numpy import pi,sqrt

model_folder="models/freq_2019_07_02_2/"

from frm_modulations import mod_list
# import tensorflow as tf
import datetime
import pickle
import sys
import copy
# In[199]:


def normalize(x):
    x=x/( np.maximum(   np.sqrt(np.mean(np.abs(x)**2)) ,1e-10 ) )
    return x.astype('complex64')


# In[200]:


def add_noise(x,snr):
    shp=np.shape(x)
    ratio = np.power(10,snr/10,dtype='float32')
    sig_pwr = np.mean(np.abs(x)**2)
    
    # There is a bug. The noise is supposed to be multiplied by sqrt(ratio) and not ratio
    y = x + sig_pwr/ratio / sqrt(2,dtype='float32') * (np.random.standard_normal(shp)  + 1j*np.random.standard_normal(shp))
    
    y = normalize(y)
    return y.astype('complex64')


# In[201]:


def generate_data(n_symbols,order):
    symbs=np.random.randint(0,order,n_symbols)
    return symbs


# In[202]:


def simulate_timing_error(x,strt_offset,step, samples):
    y = x[strt_offset:-1:step]
    y = y[:samples]
    return y


# In[203]:


def simulate_frequency_error(x,t,freq_err,phase_err):
    cf = np.cos(2*np.pi*freq_err*t+phase_err) + 1j* np.sin(2*np.pi*freq_err*t+phase_err)
    y = x*cf
    return y


# In[204]:


def simulate_realistic_channel_det(x,snr,freq_err, phase_err, fading = True,  timingErr = True):
    y = x

    if timingErr is not None:
        y = simulate_timing_error(y)

    y = simulate_frequency_error(y,freq_err,phase_err)

    if fading is not None:
        y = simulate_fading_channel(y)

    if snr is not None:
        y = add_noise(y,snr)

    return y


# In[205]:


def assign_iq(op_vec,indx, complex_vec):
    op_vec[indx,:,0] = np.real(complex_vec)
    op_vec[indx,:,1] = np.imag(complex_vec)


# In[206]:

def generate_fading_taps( symbol_time, relative_delay_spread):
    symbol_time = np.floor(symbol_time)
    delay_spread = int(np.floor(symbol_time * relative_delay_spread))
    coef2 = int(np.floor(symbol_time * relative_delay_spread/2))
    coeff = np.zeros((6,))
    coeff[3:] =  1.0 

    if delay_spread>0:  
        coeff[1] = coef2
        coeff[2] = delay_spread

        coeff[4] = 0.5 + 0.25*(2*np.random.random()-1)
        coeff[5] = 0.1+ 0.05*(2*np.random.random()-1) 
    elif coef2>0:
        coeff[1] = coef2
        coeff[2] = coef2

        t = 0.5 + 0.25*(2*np.random.random()-1)
        coeff[4] = t
        coeff[5] = t 
        
    n = np.sqrt(np.sum(np.square(coeff[3:])))
    coeff[3:]=coeff[3:]/n
    return coeff

def simulate_fading_channel(x,coeff):
    coeff_vec = np.zeros((int(coeff[2])+1,))
    coeff_vec[0] = coeff[3]
    coeff_vec[ int(coeff[1]) ] = coeff[4]
    coeff_vec[ int(coeff[2]) ] = coeff[5]
    strt_slp = x[1]-x[0]
    pad_len = int(coeff[2])
    x = np.pad(x,(pad_len,0),'linear_ramp',end_values = -strt_slp*pad_len)
    y = np.convolve(x,coeff_vec,mode='valid')
    y = y[:x.size]

    y = normalize(y)
  
    return y


def simulate_fading_channel_complex(x,coeff):
    coeff_vec = np.zeros((int(np.real(coeff[2]))+1,),dtype='complex')
    coeff_vec[0] = coeff[3]
    coeff_vec[ int(np.real(coeff[1])) ] = coeff[4]
    coeff_vec[ int( np.real(coeff[2])) ] = coeff[5]
    strt_slp = x[1]-x[0]
    pad_len = int(np.real(coeff[2]))
    x = np.pad(x,(pad_len,0),'linear_ramp',end_values = -strt_slp*pad_len)
    y = np.convolve(x,coeff_vec,mode='valid')
    y = y[:x.size]

    y = normalize(y)
  
    return y

def generate_complex_fading_taps( symbol_time, relative_delay_spread):
    # Extended pedestrian A
    # cekic_robust_2020
    # Assuming symbol rate is 10Mhz, 10
    symbol_time = np.floor(symbol_time)
    delay_spread = int(np.floor(symbol_time * relative_delay_spread))
    coef2 = int(np.floor(symbol_time * relative_delay_spread/2))
    coeff = np.zeros((6,),dtype='complex')
    coeff[3:] =  1.0 

    if delay_spread>0:  
        coeff[1] = coef2
        coeff[2] = delay_spread
        ph1 = np.random.randn()+1j*np.random.randn()
        ph1 = ph1/np.abs(ph1)
        ph2 = np.random.randn()+1j*np.random.randn()
        ph2 = ph2/np.abs(ph2)
        coeff[4] = np.random.rayleigh(0.5)*ph1
        coeff[5] = np.random.rayleigh(0.1)*ph2
    elif coef2>0:
        coeff[1] = coef2
        coeff[2] = coef2
        
        ph1 = np.random.randn()+1j*np.random.randn()
        ph1 = ph1/np.abs(ph1)
        t = np.random.rayleigh(0.5)*ph1
        coeff[4] = t
        coeff[5] = t 
        
    n = np.sqrt(np.sum(np.square(coeff[3:])))
    coeff[3:]=coeff[3:]/n
    return coeff


# In[207]:


def calc_timing_offset(mag, max_sps):
    return int(np.round(mag*max_sps))

def calc_timing_step(mag, sps, max_sps):
    return int( np.round(max_sps/ (sps*(mag+1))) )


# In[240]:

# @profile
from multiprocessing import Pool
from functools import partial

def func(my_dict):
    # print(my_dict)
    return generate_dataset_sig(**my_dict)

def generate_dataset_sig_parallel(n_samples = 300, pkt_len=2**7,
                         snr_list=np.array([30]), # np.arange(-20,5,25)
                         mod_list = mod_list,
                         same_pkt=0,
                         sps_list=[8],
                         fading = {'type': 'rand', 'mag':[0.1,0.2,0.3]},  # {'type': 'const', 'mag':0.1}, 
                         pulse_shaping = None, #{'type':'list','mag':[0.15,0.25,0.35,0.45,0.55]}
                         carrier = {'freq':{'type': 'rand', 'mag' : 10e-6}, 'phase':{'type': 'rand', 'mag' : 0}},
                         timing = {'offset':{'type': 'rand', 'mag' : 1.0}, 'symb_rate':{'type': 'rand','mag':0.25} },
                         realizations_per_sig = 1,
                         outputs = ['clean','timing','carrier','noise','fading','interm','comb'],
                         seed = None, fname = None, max_sps = 64,
                         version = 1,
                         nthreads = 10
                        ): #1e4    
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



def generate_dataset_sig(n_samples = 300, pkt_len=2**7,
                         snr_list=np.array([30]), # np.arange(-20,5,25)
                         mod_list = mod_list,
                         same_pkt=0,
                         sps_list=[8],
                         fading = {'type': 'rand', 'mag':[0.1,0.2,0.3]},  # {'type': 'const', 'mag':0.1}, 
                         pulse_shaping = None, #{'type':'list','mag':[0.15,0.25,0.35,0.45,0.55]}
                         carrier = {'freq':{'type': 'rand', 'mag' : 10e-6}, 'phase':{'type': 'rand', 'mag' : 0}},
                         timing = {'offset':{'type': 'rand', 'mag' : 1.0}, 'symb_rate':{'type': 'rand','mag':0.25} },
                         realizations_per_sig = 1,
                         outputs = ['clean','timing','carrier','noise','fading','interm','comb'],
                         seed = None, fname = None, max_sps = 64,
                         version = 1
                        ): #1e4    
    
    center_freq = 1e9
    max_sps = 64
    samp_rate = 1e6
    center_freq = 1e9
    
    args = locals()

    bl_min = True if outputs is None else False

    n_sigs = n_samples//realizations_per_sig

    if seed is not None:
            np.random.seed(seed%2**32)


    if not bl_min:
        bl_op_clean = True if 'clean' in outputs else False
        bl_op_timing = True if 'timing' in outputs else False
        bl_op_carrier = True if 'carrier' in outputs else False
        bl_op_noise = True if 'noise' in outputs else False
        bl_op_fading = True if 'fading' in outputs else False
        bl_op_interm = True if 'interm' in outputs else False
        bl_op_comb = True if 'comb' in outputs else False
        
        

        data_symbs_ideal_v = np.zeros((n_samples,pkt_len))
        
        if bl_op_clean:
            mod_symbs_v = np.zeros((n_samples,pkt_len,2))
            mod_raw_symbs_v = np.zeros((n_samples,pkt_len,2))

        if bl_op_timing:
            mod_symbs_timing_v = np.zeros((n_samples,pkt_len,2))
            mod_raw_symbs_timing_v = np.zeros((n_samples,pkt_len,2))
            mod_raw_unique_symbs_timing_v = np.zeros((n_samples,pkt_len,2))
            # mod_raw_unique_diff_symbs_timing_v = np.zeros((n_samples,pkt_len,2))
        if bl_op_carrier:
            mod_symbs_carrier_v = np.zeros((n_samples,pkt_len,2))
        if bl_op_noise:
            mod_symbs_noise_v = np.zeros((n_samples,pkt_len,2))
        if bl_op_comb:
            mod_symbs_comb_v = np.zeros((n_samples,pkt_len,2))
        if bl_op_interm:
            mod_symbs_timing_fading_v = np.zeros((n_samples,pkt_len,2))
            mod_symbs_timing_fading_freq_v = np.zeros((n_samples,pkt_len,2))
            carrier_timing_v = np.zeros((n_samples,pkt_len,2))
        if bl_op_fading:
            mod_symbs_fading_v = np.zeros((n_samples,pkt_len,2))
    else:
        mod_symbs_comb_v = np.zeros((n_samples,pkt_len,2))
        bl_op_clean = False
        bl_op_timing = False
        bl_op_carrier = False
        bl_op_noise = False
        bl_op_fading = False
        bl_op_interm = False
        bl_op_comb = True 
    
    
    modulation_v = np.zeros((n_samples,),dtype='int')
    
    
    
    
    bl_apply_carrier = False if carrier is None else True
    bl_apply_snr = False if snr_list is None else True
    bl_apply_timing = False if timing is None else True
    bl_apply_fading = False if fading is None else True
    bl_variable_pulse_shaping = False if pulse_shaping is None else True

    if bl_apply_carrier:
        freq_offset_v = np.zeros((n_samples,))
        phase_offset_v = np.zeros((n_samples,))
        carrier_v = np.zeros((n_samples,pkt_len,2))

        bl_calc_freq_loop = True if carrier['freq']['type'] == 'rand' else False
        bl_calc_phase_loop = True if carrier['phase']['type'] == 'rand' else False
        if not bl_calc_freq_loop:
            freq_err=carrier['freq']['mag']*center_freq
        if not bl_calc_phase_loop:
            phase_err=carrier['phase']['mag']
        
        
    if bl_apply_snr:
        snr_v = np.zeros((n_samples,))
        bl_calc_snr_loop = True if snr_list.size> 1 else False
        if not bl_calc_snr_loop:
            snr=snr_list[0]
    
    bl_calc_mod_loop = True if len(mod_list)> 1 else False
    if not bl_calc_mod_loop:
        mod = mod_list[0]
    bl_calc_sps_loop = True if len(sps_list)> 1 else False
    if not bl_calc_sps_loop:
        sps = sps_list[0]
    sps_v = np.zeros((n_samples,))

    if bl_apply_timing:
        data_symbs_timing_v = np.zeros((n_samples,pkt_len))
        transition_data_timing_v = np.zeros((n_samples,pkt_len))
        if version==1:
            marking_timing_v = np.zeros((n_samples,pkt_len))
        else:
            marking_b_timing_v = np.zeros((n_samples,pkt_len))


        timing_offset_v = np.zeros((n_samples,))
        symbol_rate_err_v = np.zeros((n_samples,))
        bl_calc_timing_rate_loop = True if timing['offset']['type'] == 'rand' else False
        bl_calc_symb_rate_loop = True if timing['symb_rate']['type'] == 'rand' or bl_calc_sps_loop else False
        if not bl_calc_timing_rate_loop:
            timing_offset = calc_timing_offset(timing['offset']['mag'], max_sps)
        if not bl_calc_symb_rate_loop:
            timing_step = calc_timing_step(timing['symb_rate']['mag'], sps, max_sps)

    if bl_apply_fading:
        fading_spread_v = np.zeros((n_samples,))
        fading_taps_v = np.zeros((n_samples,6))
        if fading['type'] == 'const':
            bl_calc_fading_loop = False
            fading_spread = fading['mag']
        elif  fading['type'] == 'rand':
            bl_calc_fading_loop = True
            fading_spread_list = fading['mag']

    if not bl_variable_pulse_shaping:
        pulse_ebw = 0.35
    else:
        if pulse_shaping['type']=='list':
            pulse_ebw_list = pulse_shaping['mag']
        else:
            raise NotImplemented('Only list of pulse shaping is implemented')
    pulse_ebw_v = np.zeros((n_samples,))

    strt_time = datetime.datetime.now()
    a = strt_time
    progress_step = 1000

    
    samp_indx = 0
    for sig_indx in range(n_sigs):   
        
        if bl_calc_mod_loop:
            mod = np.random.choice(mod_list)
        
        
        if mod in cont_phase_mod_list:
            order = 2
        else: # Linear modulation
            order = linear_mod_const[mod].size  
        
        if not bl_calc_sps_loop:
            sps = np.random.choice(sps_list)
        
        n_symbols = int( (pkt_len)/(sps*0.5)) + 2

        
        if same_pkt == 0:
            data_symbs=np.random.randint(0,order,n_symbols)
        elif same_pkt == 1:
            st = np.random.get_state()
            np.random.seed(5)
            data_symbs=np.random.randint(0,order,n_symbols)
            np.random.set_state(st)
        
        if bl_variable_pulse_shaping:
            pulse_ebw = np.random.choice(pulse_ebw_list)

        if not bl_min:
            pulse_ebw_v[samp_indx] = pulse_ebw
        mod_symbs_max_sps=modulate_symbols(data_symbs,mod,max_sps,ebw = pulse_ebw)
        data_symbs_max_sps= np.repeat(data_symbs,max_sps)
        
        
        t_max_sps= np.arange(0,1.0*max_sps*n_symbols/samp_rate,1.0/samp_rate)
        
        
        timing_offset_ideal =  0
        timing_step_ideal = int(max_sps / sps)
        mod_symbs_ideal = simulate_timing_error(mod_symbs_max_sps,timing_offset_ideal,timing_step_ideal, pkt_len)
        data_symbs_ideal = simulate_timing_error(data_symbs_max_sps,timing_offset_ideal,timing_step_ideal, pkt_len)
        mod_raw_symbs_ideal = modulate_symbols(data_symbs_ideal,mod,sps = 1, ebw = None, pulse_shape = None)
        transition_data_ideal = np.array(([1,]*max_sps + [0,]*max_sps) * int(n_symbols/2+1))
        t_ideal = simulate_timing_error(t_max_sps,timing_offset_ideal,timing_step_ideal, pkt_len)


        for realz_indx in range(realizations_per_sig): 
            if samp_indx%progress_step == 0:
                b = datetime.datetime.now()
                diff_time = b-a
     
                # the exact output you're looking for:
                sys.stdout.write("\rGenerated {} out of {} ({:.1f}%), Elapsed {} , estimated {}".format(samp_indx,n_samples, float(samp_indx)/n_samples*100, b-strt_time , (n_samples-samp_indx)*diff_time /progress_step ))
                sys.stdout.flush()
                a = copy.deepcopy(b)


            modulation_v[samp_indx] = mod_list.index(mod)

            if not bl_min:
                sps_v[samp_indx] = sps

            if bl_apply_timing:
                if  bl_calc_timing_rate_loop:
                    mag = np.random.random()*timing['offset']['mag']
                    timing_offset = calc_timing_offset(mag, max_sps)
                if  bl_calc_symb_rate_loop:
                    mag = (np.random.random()-0.5)*2*timing['symb_rate']['mag']
                    timing_step = calc_timing_step(mag, sps, max_sps) 
                timing_offset_v[samp_indx] = timing_offset
                symbol_rate_err_v[samp_indx] = timing_step
                mod_symbs_timing_err = simulate_timing_error(mod_symbs_max_sps,timing_offset,timing_step, pkt_len)

                data_symbs_timing_err = simulate_timing_error(data_symbs_max_sps,timing_offset,timing_step, pkt_len)
                mod_raw_symbs_timing_err = modulate_symbols(data_symbs_timing_err,mod,sps = 1, ebw = None, pulse_shape = None)

                t_timing_err = simulate_timing_error(t_max_sps,timing_offset,timing_step, pkt_len)

                marking_timing = np.repeat(np.arange(n_symbols),max_sps)
                marking_timing = simulate_timing_error(marking_timing,timing_offset,timing_step, pkt_len)

                

                

                
                # mod_raw_unique_diff_symbs_timing_err = np.diff(mod_raw_unique_symbs_timing_err,axis = 0)
                # mod_raw_unique_diff_symbs_timing_err = np.hstack((mod_raw_unique_diff_symbs_timing_err,np.zeros((pkt_len - mod_raw_unique_diff_symbs_timing_err.shape[0],))))

                # transition_data_timing = simulate_timing_error(transition_data_ideal,timing_offset,timing_step, pkt_len)
                
 
                transition_data_timing = simulate_timing_error(transition_data_ideal,timing_offset,timing_step, pkt_len+1)
                if version > 1:
                    marking_b_timing = simulate_timing_error(transition_data_ideal,timing_offset,timing_step, pkt_len)
                transition_data_timing = np.abs(np.diff(transition_data_timing)).astype('int')

                if version==2:
                    unique_marking = np.unique(marking_timing,return_index = True)[1]
                    mod_raw_unique_symbs_timing_err = mod_raw_symbs_timing_err[unique_marking]
                    mod_raw_unique_symbs_timing_err = np.hstack((mod_raw_unique_symbs_timing_err,np.zeros((pkt_len - mod_raw_unique_symbs_timing_err.shape[0],))))
                elif version==3:
                    mod_raw_unique_symbs_timing_err = mod_raw_symbs_timing_err*transition_data_timing

                if not bl_min:
                    data_symbs_timing_v[samp_indx,:] = data_symbs_timing_err
                    transition_data_timing_v[samp_indx,:] = transition_data_timing
                

                    if version ==1:
                        marking_timing_v[samp_indx,:] = marking_timing
                    else:
                        marking_b_timing_v[samp_indx,:] = marking_b_timing

                    timing_offset_v[samp_indx] = timing_offset_ideal - timing_offset
                    symbol_rate_err_v[samp_indx] = max_sps / timing_step
            else:
                mod_symbs_timing_err = mod_symbs_ideal
                t_timing_err = t_ideal
            

            if bl_apply_fading:
                if bl_calc_fading_loop:
                    fading_spread = np.random.choice(fading_spread_list)
                coeff = generate_fading_taps(max_sps / timing_step, fading_spread)
                mod_symbs_timing_fading = simulate_fading_channel(mod_symbs_timing_err, coeff)
                if not bl_min:
                    fading_taps_v[samp_indx,:] = coeff
                if bl_op_fading:
                    mod_symbs_fading = simulate_fading_channel(mod_symbs_ideal, max_sps / timing_step, fading_spread)
                fading_spread_v[samp_indx] = fading_spread
            else:
                mod_symbs_timing_fading = mod_symbs_timing_err



            if bl_apply_carrier:
                if  bl_calc_freq_loop:
                    freq_err=np.random.rand()*carrier['freq']['mag']*center_freq
                if  bl_calc_phase_loop:
                    phase_err=np.random.rand()*carrier['phase']['mag']

                if not bl_min:
                    freq_offset_v[samp_indx] = freq_err
                    phase_offset_v[samp_indx] = phase_err    

                mod_symbs_timing_fading_freq_err = simulate_frequency_error(mod_symbs_timing_fading,t_timing_err,freq_err,phase_err)
                carrier_timing_err = simulate_frequency_error(1.0,t_timing_err,freq_err,phase_err)
                if bl_op_carrier:
                    mod_symbs_carrier_err = simulate_frequency_error(mod_symbs_ideal,t_ideal,freq_err,phase_err)
                    carrier_err  = simulate_frequency_error(1.0,t_ideal,freq_err,phase_err) 
            else:
                mod_symbs_timing_fading_freq_err = mod_symbs_timing_fading
            
            
            

            
            if bl_apply_snr:
                if  bl_calc_snr_loop:
                    snr = np.random.choice(snr_list)
                    if not bl_min:
                        snr_v[samp_indx] = snr
                mod_symbs_timing_fading_freq_noise = add_noise(mod_symbs_timing_fading_freq_err,snr)
                if bl_op_noise:
                    mod_symbs_noise = add_noise(mod_symbs_ideal,snr)
            else:
                mod_symbs_timing_fading_freq_noise = mod_symbs_timing_fading_freq_err
            
            if not bl_min:
                data_symbs_ideal_v[samp_indx,:] = data_symbs_ideal
            
            if bl_op_clean:
                assign_iq(mod_symbs_v,samp_indx,mod_symbs_ideal) 
                assign_iq(mod_raw_symbs_v,samp_indx,mod_raw_symbs_ideal) 
                
            if bl_op_timing:
                assign_iq(mod_symbs_timing_v,samp_indx,mod_symbs_timing_err)
                assign_iq(mod_raw_symbs_timing_v,samp_indx,mod_raw_symbs_timing_err) 
                assign_iq(mod_raw_unique_symbs_timing_v,samp_indx,mod_raw_unique_symbs_timing_err) 
                # assign_iq(mod_raw_unique_diff_symbs_timing_v,samp_indx,mod_raw_unique_diff_symbs_timing_err) 
            if bl_op_carrier:
                assign_iq(mod_symbs_carrier_v ,samp_indx,mod_symbs_carrier_err)
                assign_iq(carrier_v,samp_indx,carrier_err)
            if bl_op_noise:
                assign_iq(mod_symbs_noise_v,samp_indx, mod_symbs_noise)
            if bl_op_fading:
                assign_iq(mod_symbs_fading_v,samp_indx, mod_symbs_fading)
            if bl_op_comb:
                assign_iq(mod_symbs_comb_v,samp_indx,mod_symbs_timing_fading_freq_noise)
            if bl_op_interm:
                assign_iq(mod_symbs_timing_fading_v,samp_indx,mod_symbs_timing_fading)
                assign_iq(carrier_timing_v,samp_indx,carrier_timing_err)
                assign_iq(mod_symbs_timing_fading_freq_v,samp_indx,mod_symbs_timing_fading_freq_err)

            samp_indx = samp_indx + 1


    op ={'sig':{},'params':{},'data':{}}
    if bl_op_clean:
        op['sig']['clean'] = mod_symbs_v
        op['sig']['raw'] = mod_raw_symbs_v
    if bl_op_timing:
        op['sig']['timing'] =  mod_symbs_timing_v
        if version==1:
            op['sig']['timing_raw'] = mod_raw_symbs_timing_v
        op['sig']['timing_raw_unique'] = mod_raw_unique_symbs_timing_v
        # op['sig']['timing_raw_unique_diff'] = mod_raw_unique_diff_symbs_timing_v  
         
    if bl_op_carrier:
        op['sig']['carrier'] =  mod_symbs_carrier_v
        op['params']['carrier'] = carrier_v
    if bl_op_noise:
        op['sig']['noise'] =  mod_symbs_noise_v
    if bl_op_fading:
        op['sig']['fading'] =  mod_symbs_fading_v
    if bl_op_comb:
        op['sig']['comb'] =  mod_symbs_comb_v
    if bl_op_interm:
        op['sig']['timing_fading_carrier'] = mod_symbs_timing_fading_freq_v
        if version==1:
            op['params']['carrier_timing'] = carrier_timing_v
        if bl_apply_timing:
            op['sig']['timing_fading'] = mod_symbs_timing_fading_v

    op['params']['mod'] = modulation_v
    if not bl_min:
        if bl_apply_fading:
            op['params']['fading_spread'] = fading_spread_v
            op['params']['fading_taps'] = fading_taps_v
        if bl_apply_carrier:
            op['params']['freq_off'] = freq_offset_v
            op['params']['phase_off'] = phase_offset_v 


        if bl_apply_timing:
            op['params']['timing_off'] = timing_offset_v
            op['params']['symb_rate'] = symbol_rate_err_v
            op['data']['timing'] = data_symbs_timing_v
            op['data']['transition'] = marking_b_timing_v
            if version==1:
                op['data']['marking'] = marking_timing_v
            else:
                op['data']['binary_marking'] = marking_b_timing_v
        

        op['params']['sps'] = sps_v
        op['params']['pulse_ebw'] = pulse_ebw_v
        if bl_apply_snr:
            op['params']['snr'] = snr_v
        if version > 1 and bl_op_clean: 
            op['data']['ideal'] = data_symbs_ideal_v

    op['args'] = args
    op['time'] = str(datetime.datetime.now())
    op['version'] = version
    if fname is not None:
        with open(fname,'wb') as f:
            pickle.dump(op,f)

    return op



def test1():
        dataset = generate_dataset_sig(n_samples = 1000 , snr_list=None,sps_list=[8],
                            timing = None, carrier = None, fading = None, outputs = {'clean'},
                            seed = 0) 
def test_data_sig_parallel():
    dataset = generate_dataset_sig_parallel(n_samples = 1000 , snr_list=np.array([0]),sps_list=[8],
                            timing = None, carrier = None, fading = None, outputs = {'clean','noise'},
                            seed = 0,fname = 'tmp/test_parallel') 
    # print(dataset)
    # print(dataset['sig']['clean'])
    print(dataset['sig']['clean'].shape)







    


if __name__ == '__main__':
    test_data_sig_parallel()