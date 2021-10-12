import numpy as np 

pkt_size = 1024
sps_rng = [4,16]
pulse_ebw_list = [0.15,0.35,0.55]
timing_offset_rng = [0,1]
fading_spread_rng = [0.0, 1]
freq_err_rng = [-0.01, 0.01]
phase_err_rng = [0, 2*np.pi]
snr_rng = [0,10]
snr_list = np.arange(snr_rng[0],snr_rng[1]+2.5,2.5)
max_sps = 64

mod_list = ['ook', 'ask4', 'ask8', 'bpsk', 'qpsk', 'psk8', 'psk16', 'psk32', 'apsk16', 'apsk32', 'apsk64', 'qam16', 'qam32', 'qam64','gmsk','cpfsk']
print(mod_list)