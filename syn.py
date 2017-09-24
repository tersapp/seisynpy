# -*- coding:utf-8 -*-
import lasio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

##########################################################
#
#       FUNCTIONS DEFINITIONS
#
def plot_vawig(axhdl, data, t, excursion, highlight=None):
    import numpy as np
    import matplotlib.pyplot as plt

    [ntrc, nsamp] = data.shape
    t = np.hstack([t.min(), t, t.max()])
    trace_scale = 10
    for i in range(0, ntrc):
        # tbuf = excursion * data[i,:] / np.max(np.abs(data)) + i
        tbuf = excursion * trace_scale * data[i, :] + i
        tbuf = np.hstack([i, tbuf, i])

        axhdl.plot(tbuf, t, color='black', linewidth=0.5)
        plt.fill_betweenx(t, tbuf, i, where=tbuf > i, facecolor=[0.8, 0, 0], linewidth=0)
        plt.fill_betweenx(t, tbuf, i, where=tbuf < i, facecolor=[0.5, 0.5, 0.5], linewidth=0)

    axhdl.set_xlim((-excursion, ntrc + excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()


def ricker(cfreq, phase, dt, wvlt_length):
    '''
    Calculate a ricker wavelet

    Usage:
    ------
    t, wvlt = wvlt_ricker(cfreq, phase, dt, wvlt_length)

    cfreq: central frequency of wavelet in Hz
    phase: wavelet phase in degrees
    dt: sample rate in seconds
    wvlt_length: length of wavelet in seconds
    '''

    import numpy as np
    import scipy.signal as signal

    nsamp = int(wvlt_length / dt + 1)
    t_max = wvlt_length * 0.5
    t_min = -t_max

    # t = np.arange(t_min, t_max, dt)

    t = np.linspace(-wvlt_length / 2, (wvlt_length - dt) / 2, wvlt_length / dt)
    wvlt = (1.0 - 2.0 * (np.pi ** 2) * (cfreq ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (cfreq ** 2) * (t ** 2))

    if phase != 0:
        phase = phase * np.pi / 180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase) * wvlt - np.sin(phase) * wvlth

    return t, wvlt


def wvlt_bpass(f1, f2, f3, f4, phase, dt, wvlt_length):
    '''
    Calculate a trapezoidal bandpass wavelet

    Usage:
    ------
    t, wvlt = wvlt_ricker(f1, f2, f3, f4, phase, dt, wvlt_length)

    f1: Low truncation frequency of wavelet in Hz
    f2: Low cut frequency of wavelet in Hz
    f3: High cut frequency of wavelet in Hz
    f4: High truncation frequency of wavelet in Hz
    phase: wavelet phase in degrees
    dt: sample rate in seconds
    wvlt_length: length of wavelet in seconds
    '''

    from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift

    nsamp = int(wvlt_length / dt + 1)

    freq = fftfreq(nsamp, dt)
    freq = fftshift(freq)
    aspec = freq * 0.0
    pspec = freq * 0.0

    # Calculate slope and y-int for low frequency ramp
    M1 = 1 / (f2 - f1)
    b1 = -M1 * f1

    # Calculate slop and y-int for high frequency ramp
    M2 = -1 / (f4 - f3)
    b2 = -M2 * f4

    # Build initial frequency and filter arrays
    freq = fftfreq(nsamp, dt)
    freq = fftshift(freq)
    filt = np.zeros(nsamp)

    # Build LF ramp
    idx = np.nonzero((np.abs(freq) >= f1) & (np.abs(freq) < f2))
    filt[idx] = M1 * np.abs(freq)[idx] + b1

    # Build central filter flat
    idx = np.nonzero((np.abs(freq) >= f2) & (np.abs(freq) <= f3))
    filt[idx] = 1.0

    # Build HF ramp
    idx = np.nonzero((np.abs(freq) > f3) & (np.abs(freq) <= f4))
    filt[idx] = M2 * np.abs(freq)[idx] + b2

    # Unshift the frequencies and convert filter to fourier coefficients
    filt2 = ifftshift(filt)
    Af = filt2 * np.exp(np.zeros(filt2.shape) * 1j)

    # Convert filter to time-domain wavelet
    wvlt = fftshift(ifft(Af))
    wvlt = np.real(wvlt)
    wvlt = wvlt / np.max(np.abs(wvlt))  # normalize wavelet by peak amplitude

    # Generate array of wavelet times
    t = np.linspace(-wvlt_length * 0.5, wvlt_length * 0.5, nsamp)

    # Apply phase rotation if desired
    if phase != 0:
        phase = phase * np.pi / 180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase) * wvlt - np.sin(phase) * wvlth

    return t, wvlt


def calc_rc(vp_mod, rho_mod):
    '''
    rc_int = calc_rc(vp_mod, rho_mod)
    '''

    nlayers = len(vp_mod)
    nint = nlayers - 1

    rc_int = []
    for i in range(0, nint):
        print(i)
        buf1 = vp_mod[i + 1] * rho_mod[i + 1] - vp_mod[i] * rho_mod[i]
        buf2 = vp_mod[i + 1] * rho_mod[i + 1] + vp_mod[i] * rho_mod[i]
        buf3 = buf1 / buf2
        rc_int.append(buf3)

    return rc_int


def calc_times(z_int, vp_mod):
    '''
    t_int = calc_times(z_int, vp_mod)
    '''

    nlayers = len(vp_mod)
    nint = nlayers - 1

    t_int = []
    for i in range(0, nint):
        if i == 0:
            tbuf = z_int[i] / vp_mod[i]
            t_int.append(tbuf)
        else:
            zdiff = z_int[i] - z_int[i - 1]
            tbuf = 2 * zdiff / vp_mod[i] + t_int[i - 1]
            t_int.append(tbuf)

    return t_int


def digitize_model(rc_int, t_int, t):
    '''
    rc = digitize_model(rc, t_int, t)

    rc = reflection coefficients corresponding to interface times
    t_int = interface times
    t = regularly sampled time series defining model sampling
    '''

    import numpy as np

    nlayers = len(rc_int)
    nint = nlayers - 1
    nsamp = len(t)

    rc = list(np.zeros(nsamp, dtype='float'))
    lyr = 0

    for i in range(0, nsamp):

        if t[i] >= t_int[lyr]:
            rc[i] = rc_int[lyr]
            lyr = lyr + 1

        if lyr > nint:
            break

    return rc

#############################################################
#岩性分析函数定义
def thick_cal(data,value,step=0.1524):
    '''
    统计列表中某数字连续出现的次数
    :param data: 1D测井岩性数据列表
    :param value: 需要统计的岩性数值
    :param step: 深度采样间隔
    :return: 返回连续出现的次数，列表
    '''
    cur_count=0
    thick_list=[]
    for idata in data:
        if idata==value:
            cur_count+=1
        else:
            if cur_count!=0:
                thick_list.append(cur_count*step)
            cur_count=0
    return thick_list
def find_pos(data,value):
    '''
    找到当前厚度值在data厚度间隔列表中的位置并且返回
    :param data: 厚度区间列表
    :param value:当前厚度数值
    :return: 厚度表中的位置
    '''
    if value<data[0]:
        return 0
    elif value>=data[-1]:
        return len(data)
    for idx in range(0,len(data)-1):
        if value>=data[idx] and value<data[idx+1]:
            return idx+1
def count_cal(data,interval=[1,5,10,15,20,25,30]):
    '''
    统计厚度列表data中的数据在interval厚度分类下出现的次数
    :param data: 厚度列表
    :param interval: 厚度间隔列表
    :return: 厚度统计数值列表
    '''
    count_list=[0 for i in range(0,len(interval)+1)]
    min_data=min(data)
    max_data=max(data)
    print('min_thick=%4.2f\tmax_thick=%4.2f'%(min_data,max_data))
    for idata in data:
        cur_pos=find_pos(interval,idata)
        count_list[cur_pos]+=1
    return count_list