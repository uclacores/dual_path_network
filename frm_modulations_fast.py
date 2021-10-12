import numpy as np

from numpy import sqrt,pi
from scipy.signal import filtfilt
from scipy.signal import convolve2d,fftconvolve
from scipy.signal import resample_poly
from scipy.signal import lfilter


import commpy
from commpy.filters import rrcosfilter,gaussianfilter

# import cv2
# from cv2 import filter2D

import matplotlib.pyplot as plt

import sys
import collections

from numba import jit



from functools import lru_cache
DEF_FFT_SIZE=256


from frm_modulations import *

def polar_to_rect(r,theta):
    return r*(np.cos(theta)+1j*np.sin(theta))

def normalize_const(symbs):
    return symbs/np.linalg.norm(symbs,2)*np.sqrt(symbs.size)
def psk_const(order, offset):
    delta = 2*pi/order
    indx = np.arange(0,order)
    phase = indx*delta+offset
    symb = polar_to_rect(1,phase)
    return normalize_const(symb)

def ask_const(order, offset):
    indx = np.arange(0,order) - (order-1)/2
    mag = indx+offset
    #symb = polar_to_rect(mag,0)
    symb = mag + 1j*0
    return normalize_const(symb)

def apsk_const(rings,offsets):
    symb = np.array([])
    for ring,offset in zip(rings,offsets):
        r = np.sin(pi/rings[0])/np.sin(pi/ring)
        delta = 2*pi/ring
        indx = np.arange(0,ring)
        phase = indx*delta+offset
        symb=np.append(symb,polar_to_rect(r,phase))
    return normalize_const(symb)

def qam_const(order):
    small_side = np.floor(np.sqrt(order))
    big_side = order/small_side
    small_indx = np.arange(small_side)-(small_side-1)/2
    big_indx = np.arange(big_side)-(big_side-1)/2
    
    xx,yy = np.meshgrid(small_indx,big_indx)
    symb = yy.flatten()+1j*xx.flatten()
    return normalize_const(symb)

linear_mod_list = ['ook','ask4','ask8','bpsk','qpsk','psk8','psk16','psk32','apsk16','apsk32','apsk64','apsk128','qam16','qam32','qam64','qam128','qam256','qam512','qam1024']
cont_phase_mod_list = ['gmsk','cpfsk']

mod_list = linear_mod_list  + cont_phase_mod_list

@lru_cache(maxsize=32)
def generate_pulse_shape_filter(sps,ebw=0.35, type='rrcos'):
    nfilts = 32
    ntaps = 11* nfilts * sps
    (t,rrc_filter) = rrcosfilter(ntaps,ebw,1,sps)
    # plt.plot(rrc_filter)
    # plt.show()
    return rrc_filter



# @profile
def linear_mod(x,mod,sps,timing_offset,timing_step,ebw=0.35,pulse_shape='rrcos'):
    const = linear_mod_const[mod]
    symbs = const[x]
    if pulse_shape is not None:
        pulse_shape_filter = generate_pulse_shape_filter(sps,ebw, pulse_shape)

        skp=int(np.floor(pulse_shape_filter.size)/2)
        strt = skp - int(sps) + timing_offset -1
        
        max_dot_len = pulse_shape_filter.size//sps
        pd_len = max_dot_len -1
        symbs_pad = np.pad(symbs,(pd_len,pd_len),mode='constant')

        # y = np.zeros(( (symbs.size*sps + pulse_shape_filter.size - sps - strt - skp)//timing_step ,),dtype='complex64')

        # print(symbs.size)


        # for i in range(strt//sps, symbs.size+pd_len - skp//sps):
        #     symbs_vec = symbs_pad[i:i+max_dot_len]
        #     for j in range(strt%sps,sps):
        #         pulse_vec = pulse_shape_filter[sps-1-j:None:sps]
        #         y[i*sps + j -strt] = np.dot(pulse_vec,symbs_vec)

        # for k in range(y.size):
        #     i = (strt+k)//sps 
        #     j = (k +strt)%sps
        #     symbs_vec = symbs_pad[i:i+max_dot_len]
        #     pulse_vec = pulse_shape_filter[sps-1-j:None:sps]
        #     y[k] = np.dot(pulse_vec,symbs_vec)

        # for k in range(y.size):
        #     r = k*timing_step
        #     i = (strt+r)//sps 
        #     j = (r +strt)%sps
        #     symbs_vec = symbs_pad[i:i+max_dot_len]
        #     pulse_vec = pulse_shape_filter[sps-1-j:None:sps]
        #     y[k] = np.dot(pulse_vec,symbs_vec)   
        y = my_interp(symbs.astype('complex64'),pulse_shape_filter.astype('complex64'),sps,timing_step,timing_offset,
                        skp,strt,max_dot_len,pd_len,symbs_pad.astype('complex64'))         


    else:
        y = symbs
    return y

@jit(nopython=True)
def my_interp(symbs,pulse_shape_filter,sps,timing_step,timing_offset,skp,strt,max_dot_len,pd_len,symbs_pad) :
    
    y = np.zeros(( (symbs.size*sps + pulse_shape_filter.size - sps - strt - skp)//timing_step ,),dtype=np.complex64)
    for k in range(y.size):
        r = k*timing_step
        i = (strt+r)//sps 
        j = (r +strt)%sps
        symbs_vec = symbs_pad[i:i+max_dot_len]
        pulse_vec = pulse_shape_filter[sps-1-j:None:sps]
        y[k] = np.dot(pulse_vec,symbs_vec)   
    return y

# @profile
def modulate_symbols_fast(x,mod,sps,timing_offset,timing_step,ebw=0.35,pulse_shape='rrcos'):

    if mod in linear_mod_list:
        y = linear_mod(x,mod,sps,timing_offset,timing_step,ebw=ebw,pulse_shape=pulse_shape)

    elif mod in cont_phase_mod_list:
        if pulse_shape is not None:
            y = cont_phase_mod(x,mod,sps,ebw=0.35,sensitivity = 1.0)
            y =y[timing_offset::timing_step]
        else:
            order = cp_mod_params[mod]['order']
            ask = ask_const(order,0.0)
            y = ask[x]
    return y




linear_mod_const ={
    'ook':ask_const(2,0.5),
    'ask4':ask_const(4,0.0),
    'ask8':ask_const(8,0.0),
    'bpsk':psk_const(2,0),
    'qpsk':psk_const(4,pi/4),
    'psk8':psk_const(8,0),
    'psk16':psk_const(16,0),
    'psk32':psk_const(32,0),
    'apsk16':apsk_const(np.array([4,12]),np.array([pi/4,0])),
    'apsk32':apsk_const(np.array([4,12,16]),np.array([0,pi/12,0])),
    'apsk64':apsk_const(np.array([4,12,20,28]),np.array([0,pi/12,0,pi/28])),
    'apsk128':apsk_const(np.array([8,16,24,36,44]),np.array([0,pi/16,0,pi/36])),
    'qam16':qam_const(16),
    'qam32':qam_const(32),
    'qam64':qam_const(64),
    'qam128':qam_const(128),
    'qam256':qam_const(256),
    'qam512':qam_const(512),
    'qam1024':qam_const(1024)

}

cp_mod_params = {
    'gmsk':{'order':2,'filter_type':'gaussian'},
    'cpfsk':{'order':2,'filter_type':'rect'},
    '4cpfsk':{'order':4,'filter_type':'rect'},
    'gfsk':{'order':2,'filter_type':'gaussian'},
    '4gfsk':{'order':4,'filter_type':'gaussian'}
}


def plot_iq(symb,show = True):
    x = np.real(symb)
    y = np.imag(symb)
    plt.plot(x)
    plt.plot(y)
    if show:
        plt.show()

if __name__ == '__main__':
    def test1():
        plot_const(psk_const(2,0))
        # plot_constellation(ask_const(4,0))
        # plot_constellation(qsk_const(4,0))
        plot_const(apsk_const(np.array([4,12]),np.array([pi/4,0])))
        
    def test2():
        for mod in linear_mod_const.keys():
            # plt.figure()
            plot_const(linear_mod_const[mod],False)
        plt.show()

    def test3():
        generate_pulse_shape_filter(8,ebw=0.35, type='rrcos')

    def test4():
        mod = 'qam1024'
        order = 1024
        n_symbols = 50
        x = np.random.randint(0,order,n_symbols)
        print(x)
        sps = 8
        y = linear_mod(x,mod,sps,ebw=0.35,pulse_shape='rrcos')
        print(y.shape)
        plot_iq(y)

    def test5():
        mod = '4gfsk'
        order = cp_mod_params[mod]['order']
        n_symbols = 10
        # x = np.random.randint(0,order,n_symbols)
        x = np.array([0,1,2,3]*10)
        print(x)
        sps = 8
        y = cont_phase_mod(x,mod, sps,0.35,sensitivity=1.0)
        print(y.shape)
        plot_iq(y)

    test4()

