# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

##########################################################
#
#       FUNCTIONS DEFINITIONS
#

def seis_fft(data ,dt ,maxfreq=100):
    '''
    采用numpy函数库中的fft函数进行傅里叶变换
    输入：
    data：一维信号，可以为ndarray
    '''
    if isinstance(data ,list): np_data=np.array (data)
    else:
        np_data=data
    spec_data=np.fft.fft(np_data)
    spec_amp=np.abs(spec_data)
    nsize=np_data.size
    freq = np.fft.fftfreq(nsize, dt)
    step = freq[1] - freq[0]
    index = int(maxfreq / step + 1)
    return freq[0:index],spec_amp[0: index]
####################################################

def cut_seismic(data,t,min_time,max_time):
    '''
    限定输出数据的范围
    返回被截断的数据
    '''
    if min_time>max_time:
        min_time=min(t)
        max_time=max(t)
        print("warning：最小时间大于最大时间，已经更改为数据的最大最小时间。")
    cut_data=[]
    cut_t=[]
    if min_time<min(t):
        min_time=min(t)
    if max_time>max(t):
        max_time=max(t)
    for index ,cur_time in enumerate(t):
        if cur_time>min_time and cur_time<max_time:
            cut_data.append(data[index])
            cut_t.append(t[index])
    return cut_data,cut_t