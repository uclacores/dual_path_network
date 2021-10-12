import pickle
import numpy as np

def load_dataset(fname,max_sps,n_mod):
    with open(fname,'rb') as f:
        dataset_test = pickle.load(f)
    
    comb_test = dataset_test['sig']['comb']    
    carrier_test = dataset_test['sig']['timing_fading_carrier']
    clean_test = dataset_test['sig']['timing']
    fading_test =  dataset_test['sig']['timing_fading']
    raw_test = dataset_test['sig']['timing_raw_unique']

    freq_raw_test = dataset_test['params']['freq_off']
    sps_test = dataset_test['params']['sps']
    timing_off_test = dataset_test['params']['timing_off']
    freq_test = (2*np.pi*freq_raw_test)
    timing_step_test = np.floor(max_sps/sps_test)
    
    timing_offNum_test =  np.round(timing_off_test*max_sps)
    coeff_test = dataset_test['params']['fading_taps']
    mod_test = dataset_test['params']['mod']
    mod_test =  to_categorical(mod_test, num_classes = n_mod)
    snr_test = dataset_test['params']['snr']
    
    op = (comb_test,carrier_test,clean_test,fading_test,raw_test,
         freq_test,timing_offNum_test,timing_step_test,
         coeff_test,mod_test,snr_test)
    
    return op
    
    

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_valid_portion(portion, comb_valid,freq_valid,fading_valid,clean_valid,timing_step_valid,
                     timing_offNum_valid, mod_valid):
    indx_valid = slice(None)
    if portion is None:
        vld_data =(comb_valid[indx_valid],
                   [freq_valid[indx_valid],fading_valid[indx_valid],
                    fading_valid[indx_valid],
                    clean_valid[indx_valid],timing_step_valid[indx_valid],
                    timing_offNum_valid[indx_valid],mod_valid[indx_valid]])
    elif portion == 1:
        vld_data =(comb_valid[indx_valid],
                   [freq_valid[indx_valid],fading_valid[indx_valid]])
    elif portion == 2:
        vld_data =(comb_valid[indx_valid],
                   [ fading_valid[indx_valid]])
    elif portion == 3:
        vld_data =(comb_valid[indx_valid],
                   [  clean_valid[indx_valid]])
    elif portion == 4:
        vld_data =(comb_valid[indx_valid],
                   [timing_step_valid[indx_valid],
                    timing_offNum_valid[indx_valid],mod_valid[indx_valid]])
    return vld_data