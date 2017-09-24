# -*- coding:utf-8 -*-
"""
此脚本用来生成叠后地震记录并分析其频谱.
输入参数：
速度、密度，如;
vp_mod = [4100.0, 4400.0, 4100.0,]  # P-wave velocity (m/s)
vs_mod = [2200.0, 2560.0, 2200.0,]  # S-wave velocity (m/s)
rho_mod= [2.6, 2.5, 2.6,]# Density (g/cc)
thickness=[5,]
Created by:    Sun Yongzhuang/Terry Sun
Create Date:   24-Sep-2017
Last Mod:

This script is provided without warranty of any kind.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from syn import *
from seisyn_fft import *
###########################################################
#
#       DEFINE MODELING PARAMETERS HERE
#
vp_mod = [4100.0, 2800.0, 4100.0, 2800.0, 4100.0, 2800.0,4100.0,]  # P-wave velocity (m/s)
vs_mod = [2200.0, 1400.0, 2200.0, 1400.0, 2200.0, 1400.0,2200.0,]  # S-wave velocity (m/s)
rho_mod= [2.6, 1.8, 2.6, 1.8, 2.6 , 1.8, 2.6,]
thickness=[1,5,2,5,1,]
str_filename='model_example.png'

#   Wavelet Parameters
wvlt_type = 'ricker'  # Valid values: 'ricker' or 'bandpass'
wvlt_length= 0.128 # Wavelet length in seconds
wvlt_phase = 0.0   # Wavelet phase in degrees
wvlt_scalar = 1.0  # Multiplier to scale wavelet amplitude (default = 1.0)
wvlt_cfreq = 32.0  # Ricker wavelet central frequency
f1 =  5.0          # Bandpass wavelet low truncation frequency
f2 = 10.0          # Bandpass wavelet low cut frequency
f3 = 50.0          # Bandpass wavelet high cut frequency
f4 = 65.0          # Bandpass wavelet high truncation frequency
isshow=1           # 1：输出到屏幕 0：输出到文件 其他：输出both
#   Trace Parameters
tmin = 0           #数据分析时窗
tmax = 0.5
dt = 0.0001 # 模型精度changing this from 0.0001 can affect the display quality

#   Plot Parameters
min_plot_time = 0.05 #绘图时窗
max_plot_time = 0.2
excursion = 2 #输出边框宽度

##########################################################
#
#       COMPUTATIONS BELOW HERE...
#
#   Generate wavelet
if wvlt_type == 'ricker':
    wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)
elif wvlt_type == 'bandpass':
    wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)
#   Apply amplitude scale factor to wavelet (to match seismic amplitude values)
wvlt_amp = wvlt_scalar * wvlt_amp
#   Calculate reflectivities from model parameters
rc_int = calc_rc(vp_mod, rho_mod)
rc_zo = []
# Calculate interface depths
temp=500.0
z_int=[500.0]
for ithick in thickness:
    temp=temp+ithick
    z_int.append(temp)
# Calculate interface times
t_int = calc_times(z_int, vp_mod)
lyr_times=t_int
# Digitize n-layer model
nsamp = int((tmax - tmin) / dt) + 1
t = []
for i in range(0, nsamp):
    t.append(i * dt)
rc = digitize_model(rc_int, t_int, t)
# seismic forward
syn_buf = np.convolve(rc, wvlt_amp, mode='same')
syn_buf =list(syn_buf)
#   限定输出范围
syn_buf,cut_t=cut_seismic(syn_buf,t,min_plot_time,max_plot_time)
syn_zo=[]
t = np.array(t)
cut_t = np.array(cut_t)
for i in range(10):
    syn_zo.append(syn_buf)
syn_zo=np.array(syn_zo)
#fft spectral analysis
spec_freq,spec_syn=seis_fft(syn_buf,dt,100)

#draw picture
fig = plt.figure(figsize=(12, 4))
fig.set_facecolor('white')
ax1=fig.add_subplot(141)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_xlabel('Syn Record')
ax1.set_ylabel('Time(ms)')
plot_vawig(ax1,syn_zo,cut_t,2)


ax3=fig.add_subplot(122)
ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')
ax3.set_xlabel('Freq(Hz)')
ax3.set_ylabel('Amp')
ax3.plot(spec_freq,spec_syn,'k',lw=2)
ax3.grid()

#   Create a "digital" time domain version of the input property model for
#   easy plotting and comparison with the time synthetic traces
nlayers = len(vp_mod)
lyr_times = np.array(lyr_times)
lyr_indx = np.array(np.round(lyr_times/dt), dtype='int16')
vp_dig = np.zeros(t.shape)
vs_dig = np.zeros(t.shape)
rho_dig = np.zeros(t.shape)
lyr1_indx = lyr_indx[0]
lyr2_indx = lyr_indx[nlayers-2]
vp_dig[0:lyr1_indx] = vp_mod[0]
vp_dig[lyr2_indx:] = vp_mod[-1]

vs_dig[0:lyr1_indx] = vs_mod[0]
vs_dig[lyr2_indx:] = vs_mod[-1]

rho_dig[0:lyr1_indx] = rho_mod[0]
rho_dig[lyr2_indx:] = rho_mod[-1]
for i in range(0,nlayers-2):
    lyr1_indx = lyr_indx[i]
    lyr2_indx = lyr_indx[i+1]
    vp_dig[lyr1_indx:lyr2_indx] = vp_mod[i+1]
    vs_dig[lyr1_indx:lyr2_indx] = vs_mod[i+1]
    rho_dig[lyr1_indx:lyr2_indx] = rho_mod[i+1]

#   Plot log curves in two-way time
ax2 = fig.add_subplot(142)
l_rho_dig, = ax2.plot(rho_dig, t, 'k', lw=2)
ax2.set_ylim((min_plot_time,max_plot_time))
ax2.set_xlim(1.8-0.03, max(rho_dig)+0.08)
ax2.invert_yaxis()
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')
ax2.set_xlabel('Den')
ax2.set_yticklabels('')
ax2.grid()


if isshow==1:
    plt.show()
elif isshow==0:
    plt.savefig(str_filename)
else:
    plt.savefig(str_filename)
    plt.show()