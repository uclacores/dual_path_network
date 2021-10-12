import pickle
import numpy as np

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(np.concatenate((data,[1])))
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)


def calc_nmse(pred,ref):

    nmse_all = np.zeros(ref.shape[0])
    for i in range(ref.shape[0]):
        yt = ref[i]
        xt = pred[i]
        xt= xt/np.sqrt(np.sum(xt**2))
        yt= yt/ np.sqrt(np.sum(yt**2))


        nmse = np.sum(xt**2)+np.sum(yt**2) - 2 * np.abs(np.sum(xt[:,0]*yt[:,0] + xt[:,1]*yt[:,1]))
        nmse_all[i] = nmse
    return nmse_all


def filter_mod_snr(vec,snr_value,mod_indx,mod_test_dec,snr_test):
    snr_test_indx_eval = snr_test == snr_value
    mod_indx  = mod_test_dec == mod_indx
    indx = np.logical_and(snr_test_indx_eval,mod_indx)
    return vec[indx]


def eval_by_snr(vec,ref,snr_test,snr_list,func):
    res_snr = np.zeros(len(snr_list))
    for i,snr_i in enumerate(snr_list):
        snr_indx = snr_test==snr_i
        res_snr[i] = func(vec[snr_indx],ref[snr_indx])
    return res_snr