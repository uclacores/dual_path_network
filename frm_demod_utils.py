import numpy as np


from frm_modulations import linear_mod_list,linear_mod_const
from commpy.filters import rrcosfilter
from scipy.signal import upfirdn
import scipy
from commpy.filters import rrcosfilter
from functools import lru_cache
from scipy.signal import filtfilt, firwin,lfilter, welch

max_sps = 64
(t,ps) = rrcosfilter(max_sps*10,0.35,1,max_sps)


def recover_symbols_params(xic,timing_offset_num,timing_step):
    us = upfirdn([1,]+[0,]*(timing_step-1),xic,timing_step)
            

    us = np.concatenate ((np.zeros(timing_offset_num,dtype='complex'),
                          us,np.zeros(5*max_sps,dtype='complex')))

    rec = scipy.signal.lfilter(ps,1, us)
    rec = rec[ps.size//2-1+max_sps:]
    rec = rec[0::max_sps]/np.sqrt(np.mean(np.abs(rec)**2))
    return rec

def get_ref_symbols(yia):
    yic = yia[:,0]+1j*yia[:,1]
    nzo= np.logical_not(np.isnan(yia[:,0]))
    yic = yic[nzo]
    yic = yic/np.max(np.abs(yic)) # FIX
    return yic

def demod_seq(xx,yy,mod_i):
    
    xx = xx/np.max(np.abs(xx))
    yy = yy/np.max(np.abs(yy))
    
    const_cmplx = linear_mod_const[mod_i]
    const_cmplx = const_cmplx/ np.max(np.abs(const_cmplx)) #* np.max(np.sqrt(np.sum(yi**2,1)))
    phase_lim = np.pi/4
    alpha = 0.5
    pred = np.zeros(min(yy.size,xx.size),dtype='int')
    bits = np.zeros(min(yy.size,xx.size),dtype='int')
    if mod_i not in ['ook']:
        phase =np.angle(np.conj(xx[0])*yy[0])
        for s_indx in range( min(yy.size,xx.size)-1):
            pred[s_indx]=np.argmin(np.abs(xx[s_indx]*np.exp(1j*phase)-const_cmplx))
            bits[s_indx]=np.argmin(np.abs(yy[s_indx]-const_cmplx))
            phase_error = np.angle(np.conj(xx[s_indx]*np.exp(1j*phase))*const_cmplx[pred[s_indx]])
            if phase_error< -phase_lim:
                phase_error = -phase_lim
            elif phase_error> phase_lim:
                phase_error = phase_lim
            phase =  phase + alpha*phase_error
#                         print(phase)
    else:
        for s_indx in range( min(yy.size,xx.size)):
            pred[s_indx]=np.argmin( np.abs(np.abs(xx[s_indx]) -const_cmplx))
            bits[s_indx]=np.argmin(np.abs(yy[s_indx]-const_cmplx))
    return (pred,bits)

def calc_ser(xic,yic,mod_i,dbg):
    ser_i = 10
    shfts = [-1,0,1] #FIXME

    for shft_sgn in shfts:

        if shft_sgn < 0:
            shft = -shft_sgn
            pred,bits = demod_seq(xic[shft:],yic,mod_i)

        else:

            shft=shft_sgn
            pred,bits = demod_seq(xic,yic[shft:],mod_i)

        ser_shft=np.mean(pred!=bits)
        if dbg:
            print(pred)
            print(bits)
            print('ser ',ser_shft)
        if  ser_shft<ser_i:
            ser_i = ser_shft
            idl_shft = shft_sgn
    return ser_i,idl_shft

def demod_batch(x,y,mod,max_sps,timing_step_list,timing_offset_list,dbg = False):
    btch = x.shape[0]
    ser = np.full(btch,np.nan)
    for i in range(btch):
        if mod[i] in linear_mod_list:
            
            timing_step = timing_step_list[i]
            timing_offset = timing_offset_list[i]

            xia = x[i,:,:]
            yia = y[i,:,:]

            xic = xia[:,0]+1j*xia[:,1]

            timing_step = int(np.round(timing_step))
            timing_offset_num = int(np.round(timing_offset))
            rec = recover_symbols_params(xic,timing_offset_num,timing_step)
            if dbg:
                print(timing_step,timing_offset_num)

            if dbg:
                pass

            
        
            xic = rec
            yic = get_ref_symbols(yia)
            
            slc = slice(0,10)
            if dbg:

                plt.figure()
                plt.plot(np.real(xic),np.imag(xic),'x')
                plt.figure()
                plt.plot(np.real(yic),np.imag(yic),'o')


            
            ser_i,idl_shft = calc_ser(xic,yic,mod[i],dbg)
            ser[i] = ser_i
            if dbg:
                print(idl_shft)
                slc = slice(0,50)
                plt.figure()
                if idl_shft <= 0:
                    plt_vec = xic[-idl_shft:]*np.exp(np.angle(np.conj(xic[0])*yic[abs(idl_shft)])*1j)
                else:
                    plt_vec= xic[:-idl_shft]*np.exp(np.angle(np.conj(xic[abs(idl_shft)])*yic[0])*1j)
                plt.plot(np.column_stack((np.real(plt_vec),np.imag(plt_vec)))[slc])
                plt.plot(np.column_stack((np.real(yic),np.imag(yic)))[slc])
                plt.legend(['pred i','pred q','ref i','ref q'])
    return ser









@lru_cache(maxsize=100)
def get_fir_filter(bw):
#     print(bw)
    return firwin(32, bw*1.5)

def frequency_noise_reduction_genie(x,freq_rad, bw):
    pkt_size = x.shape[0]
    xc = x
    xc2 = xc*np.exp(1j*np.arange(pkt_size)*(-freq_rad))
    bw_flt = np.ceil(bw*10)/10
#     print(freq_rad/(2*np.pi))
#     plt.psd(x)
    taps = get_fir_filter(bw)
    xop = np.convolve(taps,xc2,'full') 
#     print(xop.size)
    xop = xop[taps.size//2:-taps.size//2+1]
    return xop

def get_coeff_vec(coeff):
    coeff_vec = np.zeros((int(np.real(coeff[2]))+1,),dtype='complex')
    coeff_vec[0] = coeff[3]
    coeff_vec[ int(np.real(coeff[1])) ] = coeff[4]
    coeff_vec[ int( np.real(coeff[2])) ] = coeff[5]
    return coeff_vec


def mmse_equalize(xc,coeff,noise_pwr):
    coeff_vec = get_coeff_vec(coeff)
    coeff_pad = np.zeros_like(xc)
    coeff_pad[0:coeff_vec.size] = coeff_vec
    hf = np.fft.fft(coeff_pad)
    xf = np.fft.fft(xc)
    g =   xf / (hf+noise_pwr)
    gt = np.fft.ifft(g)
    return gt


def genie_demod_sample(xc,yic,mod,freq,timing_step,timing_offset,coeff,noise_pwr,dbg=False):
    if dbg:
        slc=slice(0,200)
        plt.figure()
#         plotc(xc[slc])
        plt.psd(xc)
        print(freq)
    xc = frequency_noise_reduction_genie(xc,freq*2*np.pi, timing_step/max_sps)
    if dbg:
        plt.figure()
#         plotc(xc[slc])
        plt.psd(xc)
    xc = mmse_equalize(xc,coeff,noise_pwr)
    if dbg:
        plt.figure()
        plotc(xc[slc])
    xs = recover_symbols_params(xc,int(timing_offset),int(timing_step))
    
#     xs, timing_offset = blind_symbol_recovery(xc,int(timing_step),dbg = dbg)
    
    xic = xs
    xic = xic/np.max(np.abs(xic)) 

    if dbg:
        slc = slice(0,20)
        plt.figure()
        plotc(xic[slc])
        plt.figure()
        plotc(yic[slc])
    # FIX
    if mod in linear_mod_list:
        ser_i,idl_shft = calc_ser(xic,yic,mod,dbg=dbg)
    else:
        ser_i = np.nan


    return ser_i


def genie_demod_batch(sig,freq,timing_step,timing_offset,mod,coeff,snr,correct_symbs):
    xc = sig[:,:,0]+1j*sig[:,:,1]
    
    btch_size = sig.shape[0]
    ser_op = np.ones(btch_size)
    noise_pwr = 1/(1+10**(2*snr/10))
    for i in range(btch_size):
#         print(i)
        yic = get_ref_symbols(correct_symbs[i])

        op = genie_demod_sample(xc[i],yic,mod[i],freq[i],timing_step[i],timing_offset[i],coeff[i],noise_pwr[i])
        ser_op[i] = op

    return ser_op