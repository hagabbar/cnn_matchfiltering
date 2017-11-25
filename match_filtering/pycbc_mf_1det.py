from __future__ import division
import h5py
import pickle
from gwpy.table import EventTable
import numpy as np
from scipy import integrate, interpolate
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import lal
import lalsimulation
from pylal import antenna, cosmography
import argparse
import time
from scipy.signal import filtfilt, butter
from scipy.stats import norm, chi
import os

import sys

#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm

def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='data_prep.py',
                                     description='generates GW data for application of deep learning networks.')

    # arguments for reading in a data file
    parser.add_argument('-d', '--dataset', type=str, help='test set')
    parser.add_argument('-c', '--cutoff_freq', default=12.0, type=float, help='cutoff frequency used to generate template bank')
    parser.add_argument('-tb', '--temp-bank', type=str, help='template bank .xml file')
    parser.add_argument('-f', '--fsample', type=int, default=8192, help='the sampling frequency (Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=1, help='the observation duration (sec)')
    parser.add_argument('-R', '--ROC', action='store_true', default=False,
                        help='plot ROC curve if false else save results')
    parser.add_argument('-r', '--res', type=str, default=None, help='path to file with results from CNN')
    parser.add_argument('-n', '--name', type=str, default=None, help='name for ROC plot or data')
    parser.add_argument('-I', '--detectors', type=str, nargs='+', default=['H1'], help='the detectors to use')
    parser.add_argument('-b', '--basename', type=str, default='test', help='output file path and basename.')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')
    parser.add_argument('-w', '--wave-bank', type=bool, default=False, help='waveforms already generated? (e.g. True/False')
    parser.add_argument('-wb', '--w-basename', type=str, default='test', help='location of waveform .pkl files')

    return parser.parse_args()

def tukey(M, alpha=0.5, sym=True):
    r"""Return a Tukey window, also known as a tapered cosine window.
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).
    References
    ----------
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function#Tukey_window
    Examples
    --------
    Plot the window and its frequency response:
    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt
    >>> window = signal.tukey(51)
    >>> plt.plot(window)
    >>> plt.title("Tukey window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.ylim([0, 1.1])
    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Tukey window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    """
    if _len_guards(M):
        return np.ones(M)

    if alpha <= 0:
        return np.ones(M, 'd')
    elif alpha >= 1.0:
        return hann(M, sym=sym)

    M, needs_trunc = _extend(M, sym)

    n = np.arange(0, M)
    width = int(np.floor(alpha * (M - 1) / 2.0))
    n1 = n[0:width + 1]
    n2 = n[width + 1:M - width - 1]
    n3 = n[M - width - 1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (M - 1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1))))

    w = np.concatenate((w1, w2, w3))

    return _truncate(w, needs_trunc)

def inner_alt(a, b, T_obs, fs, psd):
    """
    Computes the noise weighted inner product in the frequency domain
    Follows Babak et al Eq. 2 where one product is whitened and 
    the other is unwhitned.
    """
    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs

    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    af = np.fft.rfft(a * win) * dt
    bf = np.fft.rfft(b * win) * dt
    temp = 4.0 * np.real(np.sum((np.conj(af) * bf) / np.sqrt(psd))) * df
    return temp

def inner_FD(a, b, T_obs, fs, psd):
    """
    Computes the noise weighted inner product in the frequency domain
    Follows Babak et al Eq. 2 assuming both products are unwhitened.
    """
    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs

    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    af = a * dt
    bf = b * dt # originally multiplied dt by np.fft.rfft(b * win)

    temp = 4.0 * np.real(np.sum((np.conj(af) * bf) / psd)) * df # was originally complex conjugate of af
    return temp

def inner(a, b, T_obs, fs, psd):
    """
    Computes the noise weighted inner product in the frequency domain
    Follows Babak et al Eq. 2 assuming both products are unwhitened.
    """
    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs

    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    af = np.fft.rfft(a * win) * dt
    bf = b * dt # originally multiplied dt by np.fft.rfft(b * win)

    temp = 4.0 * np.real(np.sum((np.conj(af) * bf) / psd)) * df # was originally complex conjugate of af
    return temp

def meas_snr(data, template_p, template_c, Tobs, fs, psd):
    """
    Computes the measured SNR for a given template and dataset
    Follows Babak et al Eq. 9
    """

    a = inner(data, template_p, Tobs, fs, psd)
    b = inner(data, template_c * 1.j, Tobs, fs, psd)
    c = inner_FD(template_p, template_p, Tobs, fs, psd)
    #print data, whiten_data(data,psd,fs)
    #sys.exit()

    return np.sqrt((a * a + b * b) / c)

def whiten_data(data,duration,sample_rate,psd):
    """
    Takes an input timeseries and whitens it according to a psd
    """

    # FT the input timeseries
    #win = tukey(duration*sample_rate,alpha=1.0/8.0)
    xf = data.real #np.fft.rfft(data)

    # deal with undefined PDS bins and normalise
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300
    xf /= (np.sqrt(0.5*psd*sample_rate))

    # Detrend the data: no DC component.
    xf[0] = 0.0

    # Return to time domain.
    #x = np.fft.irfft(xf)

    # Done.
    return xf

def whiten_data_losc(data, psd, fs):
    """
    Whitens the data
    Based on the LOSC tutorial code
    """    

    Nt = len(data)
    dt = 1.0/fs
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(data)
    white_hf = hf / (np.sqrt(psd /dt/2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht 

def looper(sig_data,tmp_bank,T_obs,fs,dets,psds,wpsds,basename,w_basename,f_low=12.0,wave_bank=False):

    # define input parameters
    N = T_obs * fs  # the total number of time samples
    dt = 1 / fs  # the sampling time (sec)
    amplitude_order = 0
    phase_order = 7
    approximant = lalsimulation.IMRPhenomD # waveform
    ndet = len(dets)  # number of detectors
    dist = 1e6 * lal.PC_SI  # put it as 1 MPc

    # make waveforms for template bank
    if wave_bank == False:
        # loop over template bank params
        for idx,w in enumerate(tmp_bank):
           if idx == 0:
               hp,hc,fmin = make_waveforms(w,dt,dist,fs,approximant,N,ndet,dets,psds,T_obs,f_low)
               hp_bank = {idx:hp}
               hc_bank = {idx:hc}
               fmin_bank = {idx:fmin}
           #if idx == 10:
           #    break
           else:
               hp_new,hc_new,fmin_new = make_waveforms(w,dt,dist,fs,approximant,N,ndet,dets,psds,T_obs,f_low)
               hp_bank.update({idx:hp_new})
               hc_bank.update({idx:hc_new})
               fmin_bank.update({idx:fmin_new})

        # dump contents of hp and hc banks to pickle file
        pickle_hp = open("%shp.pkl" % basename,"wb")
        pickle.dump(hp_bank, pickle_hp)
        pickle_hp.close()
        pickle_hc = open("%shc.pkl" % basename,"wb")
        pickle.dump(hc_bank, pickle_hc)
        pickle_hc.close()
        pickle_fmin = open("%sfmin.pkl" % basename,"wb")
        pickle.dump(fmin_bank, pickle_fmin)
        pickle_fmin.close()

        hp = hp_bank
        hc = hc_bank

    # load waveforms if already made
    else:
        # load hplus and hcross pickle file
        pickle_hp = open("%shp.pkl" % w_basename,"rb")
        hp = pickle.load(pickle_hp)
        pickle_hc = open("%shc.pkl" % w_basename,"rb")
        hc = pickle.load(pickle_hc)
        pickle_fmin = open("%sfmin.pkl" % w_basename,"rb")
        fmin_bank = pickle.load(pickle_fmin)

    # loop over test signals
    # not setup to do multi detector network yet
    # If you're reading this code, I'm sorry but ...
    # welcome to the 7th circle of hell.
    for det,psd,wpsd in zip(dets,psds,wpsds):
        sig_match_rho = []
        hp_hc_wvidx = []
        chi_rho = []
        noise = sig_data[0][sig_data[1]==0]
        signal = sig_data[0][sig_data[1]==1]

        chi_bool = True
        if chi_bool == True:
            #psd_wht = gen_psd(fs, 1, op='AdvDesign', det='H1')
            count = 0
            for idx in xrange(sig_data[0].shape[0]):
                if sig_data[1][idx] == 0:
                    # whitened first template
                    h_idx = random.choice(hp.keys())
                    #hp_1_wht = chris_whiten_data(hp[h_idx], T_obs, fs, psd.data.data, flag='fd')
                    #hc_1_wht = chris_whiten_data(hc[h_idx], T_obs, fs, psd.data.data, flag='fd')

                    # calculate chi distribution. For testing purposes only!
                    #chi_rho.append(meas_snr(sig_data[0][idx][0], hp_1_wht, hc_1_wht, T_obs, fs, wpsd))
                    chi_rho.append(chris_snr_ts(sig_data[0][idx],hp[h_idx],hc[h_idx],T_obs,fs,wpsd,fmin_bank[h_idx],flag='fd')[0][int(N/2)])
                    count+=1
                    print '{}: Chi Rho for signal {} = {}'.format(time.asctime(),idx,chi_rho[-1])

            # save list of chi rho for test purposes only
            print np.mean(chi_rho), np.std(chi_rho)
            pickle_out = open("%schirho_values.pickle" % basename, "wb")
            pickle.dump(chi_rho, pickle_out)
            pickle_out.close()
            
        # this loop defines how many signals you are looping over
        #psd_wht = gen_psd(fs, 5, op='AdvDesign', det='H1')
        #for i in xrange(sig_data[0].shape[0]):
        for i in range(1000):
            rho = -np.inf

            for j, M in enumerate(hp):
                # compute the max(SNR) of this template
                #hp_1_wht = chris_whiten_data(hp[j], T_obs, fs, psd.data.data, flag='fd')
                #hc_1_wht = chris_whiten_data(hc[j], T_obs, fs, psd.data.data, flag='fd')
                #max_rho = max(snr_ts(sig_data[0][i],hp_1_wht,hc_1_wht,T_obs,fs,wpsd)[0])
                max_rho = max(chris_snr_ts(sig_data[0][i],hp[j],hc[j],T_obs,fs,wpsd,fmin_bank[j],flag='fd')[0][int(fs*1.245):int(fs*1.455)]) #had [0] here
                # check if max(SNR) greater than rho
                if max_rho > rho:
                    rho = max_rho
                    #hphcidx = j
                    #hphcidx = [hp_new,hc_new]
            print '{}: Max(rho) for signal {} type {} = {}'.format(time.asctime(),i,sig_data[1][i],rho)
            
            # store max snr and index of hp/hc waveforms
            sig_match_rho.append(rho)
        print np.mean(sig_match_rho), np.std(sig_match_rho)
        #hp_hc_wvidx.append(hphcidx)



    return np.array(sig_match_rho), np.array(chi_rho)

def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1

def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False

def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w

def get_snr(data,T_obs,fs,psd):
    """
    computes the snr of a signal in unit variance time domain noise
    """

    N = T_obs*fs
    df = 1.0/T_obs
    dt = 1.0/fs

    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300

    xf = np.fft.rfft(data)*dt
    SNRsq = 4.0*np.sum((np.abs(xf)**2)/psd)*df
    return np.sqrt(SNRsq)


def chris_whiten_data(data,duration,sample_rate,psd,flag='td'):
    """
    Takes an input timeseries and whitens it according to a psd
    """

    if flag=='td':
        # FT the input timeseries - window first
        win = tukey(duration*sample_rate,alpha=1.0/8.0)
        xf = np.fft.rfft(win*data)
    else:
        xf = data

    # deal with undefined PDS bins and normalise
    #idx = np.argwhere(psd>0.0)
    #invpsd = np.zeros(psd.size)
    #invpsd[idx] = 1.0/psd[idx]
    #xf *= np.sqrt(2.0*invpsd/sample_rate)

    # deal with undefined PDS bins and normalise
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300
    xf /= (np.sqrt(0.5*psd*sample_rate))

    # Detrend the data: no DC component.
    xf[0] = 0.0

    if flag=='td':
        # Return to time domain.
        x = np.fft.irfft(xf)
        return x
    else:
        return xf

def chris_snr_ts(data,template_p,template_c,Tobs,fs,psd,fmin,flag='td'):
    """
    Computes the SNR timeseries given a timeseries and template
    """
    N = Tobs*fs
    df = 1.0/Tobs
    dt = 1.0/fs
    fidx = int(fmin/df)
    win = tukey(N,alpha=1.0/8.0)

    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300
    freqs = np.fft.fftfreq(N,dt)
    oldfreqs = df*np.arange(N//2 + 1)
    intpsd = np.interp(np.abs(freqs),oldfreqs,psd)
    idx = np.argwhere(intpsd==0.0)
    intpsd[idx] = 1e300
    idx = np.argwhere(np.isnan(intpsd))
    intpsd[idx] = 1e300

    if flag=='td':
        # make complex template
        temp = template_p + template_c*1.j
        ftemp = np.fft.fft(temp*win)*dt
    else:
        # same as fft(temp_p) + i*fft(temp_c)
        temp_p = np.hstack([template_p,np.conj((template_p[::-1])[1:-1])])
        temp_c = np.hstack([template_c,np.conj((template_c[::-1])[1:-1])])
        ftemp = temp_p + 1.j*temp_c
        # fill negative frequencies - only set up to do N=even
        #rev = temp[::-1]
        #ftemp = np.hstack([temp,np.conj(rev[1:-1])])
    ftemp[:fidx] = 0.0
    ftemp[-fidx:] = 0.0

    # FFT data
    fdata = np.fft.fft(data*win)*dt

    z = 4.0*np.fft.ifft(fdata*np.conj(ftemp)/intpsd)*df*N
    s = 4.0*np.sum(np.abs(ftemp, dtype='float128')**2/intpsd)*df
    return np.abs(z)/np.sqrt(s)

def snr_ts(data, template_p, template_c, Tobs, fs, psd):
    """
    Computes the SNR for each time step
    Based on the LOSC tutorial code
    """

    Nyq = fs / 2.
    N = Tobs * fs
    N_nyq = Tobs * Nyq
    df = 1.0 / Tobs
    dt = 1.0 / fs
    dt_nyq = 1.0 / Nyq

    temp = template_p + template_c * 1.j # didn't have dt before
    dwindow = tukey(N, alpha=1.0 / 8.0)
    # dwindow = np.ones(temp.size)

    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data * dwindow) * dt
    #template_fft = np.fft.fft(temp * dwindow) * dt

    # use nyquist for fs
    freqs = np.fft.fftfreq(N, dt)
    oldfreqs = df * np.arange(N // 2 + 1)

    intpsd = np.interp(np.abs(freqs), oldfreqs, psd)
    idx = np.argwhere(intpsd == 0.0)
    intpsd[idx] = 1e300
    idx = np.argwhere(np.isnan(intpsd))
    intpsd[idx] = 1e300

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    print temp.conjugate().shape
    sys.exit()
    print data_fft.shape, temp.conjugate().shape, intpsd.shape
    sys.exit()
    optimal = data_fft * temp.conjugate() / intpsd # used to be template_fft.conj()
    optimal_time = 2 * np.fft.ifft(optimal) * fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1 * (temp * temp.conjugate() / intpsd).sum() * df # used to be template_fft.conj() and template_fft
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time / sigma

    return abs(SNR_complex)


def get_fmin(mc,dt):
    """
    Compute the instantaneous frequency given a chirp mass (in Msun) and time till merger in seconds
    """

    # convert to SI c=G=1 units
    mcsi = mc*lal.MSUN_SI*lal.G_SI/lal.C_SI**3
    fmin = (1.0/(8.0*np.pi)) * ((0.2*dt)**(-3.0/8.0)) * (mcsi**(-5.0/8.0))
    print '{}: signal enters segment at {} Hz'.format(time.asctime(),fmin)
    return fmin


def make_waveforms(template,dt,dist,fs,approximant,N,ndet,dets,psds,T_obs,f_low=12.0):
    """ make waveform"""


    # define variables
    template = list(template)
    m12 = [template[0],template[1]]
    eta = template[2]
    mc = template[3]
    N = T_obs * fs      # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    approximant = lalsimulation.IMRPhenomD
    f_high = fs/2.0
    df = 1.0/T_obs
    f_low = df*int(get_fmin(mc,1.0)/df)
    f_ref = f_low    
    dist = 1e6*lal.PC_SI  # put it as 1 MPc

    # generate iota
    iota = np.arccos(-1.0 + 2.0*np.random.rand())
    print '{}: selected bbh cos(inclination) = {}'.format(time.asctime(),np.cos(iota))

    # generate polarisation angle
    psi = 2.0*np.pi*np.random.rand()
    print '{}: selected bbh polarisation = {}'.format(time.asctime(),psi)

    # print parameters
    print '{}: selected bbh mass 1 = {}'.format(time.asctime(),m12[0])
    print '{}: selected bbh mass 2 = {}'.format(time.asctime(),m12[1])
    print '{}: selected bbh eta = {}'.format(time.asctime(),eta)

    # make waveform
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
                    m12[0] * lal.MSUN_SI, m12[1] * lal.MSUN_SI,
                    0, 0, 0, 0, 0, 0,
                    dist,
                    iota, 
                    0, 0, 0, 0,
                    df,
                    f_low,f_high,
                    f_ref,
                    lal.CreateDict(),
                    approximant)

    # define variables
    #template = list(template)
    #m12 = [template[0],template[1]]
    #eta = template[2]
    #mc = template[3]
    #Nyq = fs #/ 2. - 1
    #f_low = get_fmin(mc,T_obs)


    # loop until we have a long enough waveform - slowly reduce flow is needed
    #flag = False
    #while not flag:
    #hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
    #            m12[0] * lal.MSUN_SI, m12[1] * lal.MSUN_SI,
    #            0, 0, 0, 0, 0, 0,
    #            dist,
    #            iota, 0, 0,
    #            0, 0,
    #            T_obs,
    #            f_low, Nyq, f_low,
    #            lal.CreateDict(),
    #            approximant)
        #flag = True if hp.data.length>3*N else False
        #f_low -= 1       # decrease by 1 Hz each time


    hp = hp.data.data
    hc = hc.data.data
    for psd in psds:
        hp_1_wht = chris_whiten_data(hp, T_obs, fs, psd.data.data, flag='fd')
        hc_1_wht = chris_whiten_data(hc, T_obs, fs, psd.data.data, flag='fd')


    return hp_1_wht,hc_1_wht,get_fmin(mc,1)

def gen_psd(fs, T_obs, op='AdvDesign', det='H1'):
    """
    generates noise for a variety of different detectors
    """
    N = T_obs * fs  # the total number of time samples
    dt = 1 / fs  # the sampling time (sec)
    df = 1 / T_obs  # the frequency resolution
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df, lal.HertzUnit, N // 2 + 1)

    if det == 'H1' or det == 'L1':
        if op == 'AdvDesign':
            lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyLow':
            lalsimulation.SimNoisePSDAdVEarlyLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyHigh':
            lalsimulation.SimNoisePSDAdVEarlyHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidLow':
            lalsimulation.SimNoisePSDAdVMidLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidHigh':
            lalsimulation.SimNoisePSDAdVMidHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateLow':
            lalsimulation.SimNoisePSDAdVLateLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateHigh':
            lalsimulation.SimNoisePSDAdVLateHighSensitivityP1200087(psd, 10.0)
        else:
            print 'unknown noise option'
            exit(1)
    else:
        print 'unknown detector - will add Virgo soon'
        exit(1)

    return psd

def load_data(initial_dataset):
    # get core name of dataset
    #name1 = initial_dataset.split('_0')[0]
    #name2 = initial_dataset.split('_0')[1]
    print('Using data for: {0}'.format(initial_dataset))  
     
    #load in dataset 0
    with open(initial_dataset, 'rb') as rfp:
        base_test_set = pickle.load(rfp)

    return base_test_set

def main():
    # get the command line args
    args = parser()
    np.random.seed(args.seed)

    # set path to file
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath(args.dataset, cur_path)
     
    # load dataset
    data = load_data(new_path)

    # redefine things for conciseness
    Tobs = args.Tobs  # observation time
    fs = args.fsample  # sampling frequency
    dets = args.detectors  # detectors
    ndet = len(dets)  # number of detectors
    N = Tobs * fs  # the total number of time samples
    n = N // 2 + 1  # the number of frequency bins
    tmp_bank = args.temp_bank # template bank file
    f_low = args.cutoff_freq # cutoff frequency used in template generation

    psds = [gen_psd(fs, Tobs, op='AdvDesign', det=d) for d in args.detectors]
    wpsds = (2.0 / fs) * np.ones((ndet, n))  # define effective PSD for whited data

    # load template bank
    tmp_bank = np.array(EventTable.read(tmp_bank,
    format='ligolw.sngl_inspiral', columns=['mass1','mass2','eta','mchirp']))

    # loop over stuff
    output,chi_test = looper(data,tmp_bank,Tobs,fs,dets,psds,wpsds,args.basename,args.w_basename,args.cutoff_freq,args.wave_bank)
    chi_test = [chi_test,data[1]]
    output = [output,data[1]]

    # save list of rho for test signals and test noise
    pickle_out = open("%srho_values.pickle" % args.basename, "wb")
    pickle.dump(output, pickle_out)
    pickle_out.close()

    # save list of chi rho for test purposes only
    pickle_out = open("%schirho_values.pickle" % args.basename, "wb")
    pickle.dump(chi_test, pickle_out)
    pickle_out.close()

if __name__ == "__main__":
    main()
