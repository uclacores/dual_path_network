import numpy as np
from keras.utils  import Sequence
from frm_dataset_creator2 import create_sample_fast

class train_generator(Sequence):
    'Generates data for Keras'
    def __init__(self, samples_per_epoch, batch_size,pkt_size,max_sps,mod_list,
                   sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng):
        self.first_run = True
        
        self.batch_size = batch_size 
        self.pkt_size = pkt_size
        self.max_sps = max_sps
        self.mod_list = mod_list
        self.sps_rng = sps_rng
        self.pulse_ebw_list = pulse_ebw_list
        self.timing_offset_rng = timing_offset_rng
        self.fading_spread_rng = fading_spread_rng
        self.freq_err_rng = freq_err_rng
        self.phase_err_rng = phase_err_rng
        self.snr_rng = snr_rng
        
        self.samples_per_epoch = samples_per_epoch
        self.n_mods = len(mod_list)


        
        
    def __len__(self):
          return self.samples_per_epoch  // self.batch_size

    def __getitem__(self, index):
        if self.first_run:
            np.random.seed()
            self.first_run = False
            
        batch_size = self.batch_size
        pkt_size = self.pkt_size
        mod_list = self.mod_list
        max_sps = self.max_sps
        
        comb_v = np.zeros((batch_size,pkt_size,2))
        carrier_v = np.zeros((batch_size,pkt_size,2))
        fading_v = np.zeros((batch_size,pkt_size,2))
        clean_v = np.zeros((batch_size,pkt_size,2))
        timing_v = np.zeros((batch_size,pkt_size,2))
        raw_v =  np.zeros((batch_size,pkt_size,2))
        mod_v = np.zeros((batch_size,len(mod_list)))
        samp_weights_v = np.zeros((batch_size,))
        
        mod_index = np.random.choice(self.n_mods,(batch_size,)).astype(np.int_)
        mod_v[range(batch_size),mod_index] = 1
        sps = np.random.uniform(self.sps_rng[0],self.sps_rng[1],(batch_size,))
        pulse_ebw = np.random.choice(self.pulse_ebw_list,(batch_size,))
        timing_offset = np.random.uniform(self.timing_offset_rng[0],self.timing_offset_rng[1],(batch_size,))
        fading_spread = np.random.uniform(self.fading_spread_rng[0],self.fading_spread_rng[1],(batch_size,))
        freq_err = np.random.uniform(self.freq_err_rng[0],self.freq_err_rng[1],(batch_size,))
        phase_err = np.random.uniform(self.phase_err_rng[0],self.phase_err_rng[1],(batch_size,))
        snr = np.random.uniform(self.snr_rng[0],self.snr_rng[1],(batch_size,))
        
        if 'cpfsk' in mod_list:
            cpfsk_loc =  mod_index == mod_list.index('cpfsk')
        else:
            cpfsk_loc = np.zeros_like(mod_index,dtype='bool')
        if 'gmsk' in mod_list:
            gmsk_loc =  mod_index == mod_list.index('gmsk')
        else:
            gmsk_loc = np.zeros_like(mod_index,dtype='bool')
        
        non_linear_mod = np.logical_or( gmsk_loc,cpfsk_loc) 
        
        samp_weights_v = 1-np.exp(-0.4 *10**((snr)/10))
        

        freq_v = (2*np.pi*freq_err)
        
        for i in range(batch_size):
            mod = self.mod_list[mod_index[i]]
            op = create_sample_fast( mod = mod,pkt_len = self.pkt_size,sps=sps[i],pulse_ebw = pulse_ebw[i],
                  timing_offset = timing_offset[i], 
                  fading_spread = fading_spread[i], 
                  freq_err = freq_err[i], phase_err =phase_err[i],    
                  snr = snr[i], max_sps = self.max_sps,
                  complex_fading=True, freq_in_hz = True,
                  seed = None)
            comb_v[i] ,carrier_v[i],fading_v[i],clean_v[i],timing_v[i],raw_v[i],coeff = op
        
        timing_step_v = np.floor(max_sps/sps)
        timing_offNum_v =  np.round(timing_offset*max_sps)

        return ([comb_v], [freq_v,fading_v,fading_v,clean_v,timing_step_v,timing_offNum_v,mod_v], 
                [np.ones((batch_size,)),np.ones((batch_size,)),samp_weights_v,
                 samp_weights_v,samp_weights_v * np.logical_not(non_linear_mod),
                 samp_weights_v * np.logical_not(non_linear_mod),np.ones((batch_size,))])

    
class train_generator_mod(Sequence):
    'Generates data for Keras'
    def __init__(self, samples_per_epoch, batch_size,pkt_size,max_sps,mod_list,
                   sps_rng,pulse_ebw_list,timing_offset_rng,fading_spread_rng,freq_err_rng,phase_err_rng,snr_rng):
        self.first_run = True
        
        self.batch_size = batch_size 
        self.pkt_size = pkt_size
        self.max_sps = max_sps
        self.mod_list = mod_list
        self.sps_rng = sps_rng
        self.pulse_ebw_list = pulse_ebw_list
        self.timing_offset_rng = timing_offset_rng
        self.fading_spread_rng = fading_spread_rng
        self.freq_err_rng = freq_err_rng
        self.phase_err_rng = phase_err_rng
        self.snr_rng = snr_rng
        
        self.samples_per_epoch = samples_per_epoch
        
        self.n_mods = len(mod_list)


        
        
    def __len__(self):
          return self.samples_per_epoch  // self.batch_size

    def __getitem__(self, index):
        if self.first_run:
            np.random.seed()
            self.first_run = False
        batch_size = self.batch_size
        pkt_size = self.pkt_size
        max_sps = self.max_sps
        mod_list = self.mod_list
        
        comb_v = np.zeros((batch_size,pkt_size,2))
        carrier_v = np.zeros((batch_size,pkt_size,2))
        fading_v = np.zeros((batch_size,pkt_size,2))
        clean_v = np.zeros((batch_size,pkt_size,2))
        timing_v = np.zeros((batch_size,pkt_size,2))
        raw_v =  np.zeros((batch_size,pkt_size,2))
        mod_v = np.zeros((batch_size,len(mod_list)))
        
        mod_index = np.random.choice(self.n_mods,(batch_size,)).astype(np.int_)
        mod_v[range(batch_size),mod_index] = 1
        sps = np.random.uniform(self.sps_rng[0],self.sps_rng[1],(batch_size,))
        pulse_ebw = np.random.choice(self.pulse_ebw_list,(batch_size,))
        timing_offset = np.random.uniform(self.timing_offset_rng[0],self.timing_offset_rng[1],(batch_size,))
        fading_spread = np.random.uniform(self.fading_spread_rng[0],self.fading_spread_rng[1],(batch_size,))
        freq_err = np.random.uniform(self.freq_err_rng[0],self.freq_err_rng[1],(batch_size,))
        phase_err = np.random.uniform(self.phase_err_rng[0],self.phase_err_rng[1],(batch_size,))
        snr = np.random.uniform(self.snr_rng[0],self.snr_rng[1],(batch_size,))
        
        
        for i in range(batch_size):
            mod = self.mod_list[mod_index[i]]
            op = create_sample_fast( mod = mod,pkt_len = self.pkt_size,sps=sps[i],pulse_ebw = pulse_ebw[i],
                  timing_offset = timing_offset[i], 
                  fading_spread = fading_spread[i], 
                  freq_err = freq_err[i], phase_err =phase_err[i],    
                  snr = snr[i], max_sps = self.max_sps,
                complex_fading=True, freq_in_hz = True,
                  seed = None)
            comb_v[i] ,carrier_v[i],fading_v[i],clean_v[i],timing_v[i],raw_v[i],coeff = op

        return [comb_v], [mod_v]
    
  