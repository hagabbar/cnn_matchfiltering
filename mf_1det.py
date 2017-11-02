#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import numpy as np
from scipy import integrate, interpolate
from scipy.misc import imsave
from gwpy.table import EventTable

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

#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm


class bbhparams:
    def __init__(self,mc,m1,m2,ra,dec,cosi,psi,tc,snr,SNR):
        self.mc = mc
        self.m1 = m1
        self.m2 = m2
        self.ra = ra
        self.dec = dec
        self.cosi = psi
        self.tc = tc
        self.snr = snr
        self.SNR = SNR


class ROCresults:
    def __init__(self,TP,FN,FP,TN,thresholds,Nsignal,Nnoise,SNR):
        self.TP = TP
        self.FN = FN
        self.FP = FP
        self.TN = TN
        self.thresholds = thresholds
        self.Nsignal = Nsignal
        self.Nnoise = Nnoise
        self.SNR = SNR


def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='data_prep.py',
                                     description='generates GW data for application of deep learning networks.')

    # arguments for reading in a data file
    parser.add_argument('-N', '--Nsig', type=int, default=3000, help='the number of example signals')
    parser.add_argument('c', '--cutoff_freq', type=int, help='cutoff frequency used to generate template bank')
    parser.add_argument('-tb', '--temp-bank', type=str, help='template bank .xml file')
    parser.add_argument('-f', '--fsample', type=int, default=8192, help='the sampling frequency (Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=1, help='the observation duration (sec)')
    parser.add_argument('-s', '--isnr', type=float, default=8, help='the signal integrated SNR')
    parser.add_argument('-R', '--ROC', action='store_true', default=False,
                        help='plot ROC curve if false else save results')
    parser.add_argument('-r', '--res', type=str, default=None, help='path to file with results from CNN')
    parser.add_argument('-n', '--name', type=str, default=None, help='name for ROC plot or data')
    parser.add_argument('-I', '--detectors', type=str, nargs='+', default=['H1'], help='the detectors to use')
    parser.add_argument('-b', '--basename', type=str, default='test', help='output file path and basename')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')

    return parser.parse_args()


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

def gen_noise(fs, T_obs, psd):
    """
    Generates noise from a psd
    """

    N = T_obs * fs  # the total number of time samples
    Nf = N // 2 + 1
    dt = 1 / fs  # the sampling time (sec)
    df = 1 / T_obs

    amp = np.sqrt(0.25 * T_obs * psd)
    idx = np.argwhere(psd == 0.0)
    amp[idx] = 0.0
    idx = np.argwhere(psd == 1e300)
    amp[idx] = 0.0
    re = amp * np.random.normal(0, 1, Nf)
    im = amp * np.random.normal(0, 1, Nf)
    re[0] = 0.0
    im[0] = 0.0
    x = N * np.fft.irfft(re + 1j * im) * df

    return x


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


def get_snr(data, T_obs, fs, psd):
    """
    computes the snr of a whitened signal in unit variance time domain noise
    """

    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs

    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    xf = np.fft.rfft(data * win) * dt
    SNRsq = 4.0 * np.sum((np.abs(xf) ** 2) / psd) * df
    return np.sqrt(SNRsq)


def inner(a, b, T_obs, fs, psd):
    """
    Computes the noise weighted inner product in the frequency domain
    Follows Babak et al Eq. 2
    """
    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs

    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    af = np.fft.rfft(a * win) * dt
    bf = np.fft.rfft(b * win) * dt
    temp = 4.0 * np.real(np.sum((np.conj(af) * bf) / psd)) * df
    return temp


def meas_snr(data, template_p, template_c, Tobs, fs, psd):
    """
    Computes the measured SNR for a given template and dataset
    Follows Babak et al Eq. 9
    """

    a = inner(data, template_p, Tobs, fs, psd)
    b = inner(data, template_c, Tobs, fs, psd)
    c = inner(template_p, template_p, Tobs, fs, psd)
    return np.sqrt((a * a + b * b) / c)


def snr_ts(data, template_p, template_c, Tobs, fs, psd):
    """
    Computes the SNR for each time step
    Based on the LOSC tutorial code
    """

    N = Tobs * fs
    df = 1.0 / Tobs
    dt = 1.0 / fs

    temp = template_p + template_c * 1.j
    dwindow = tukey(temp.size, alpha=1.0 / 16.0)
    # dwindow = np.ones(temp.size)

    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data * dwindow) * dt
    template_fft = np.fft.fft(temp * dwindow ) * dt
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
    optimal = data_fft * template_fft.conjugate() / intpsd
    optimal_time = 2 * np.fft.ifft(optimal) * fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1 * (template_fft * template_fft.conjugate() / intpsd).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time / sigma

    return abs(SNR_complex)


def whiten_data(data, psd, fs):
    """
    Whitens the data
    Based on the LOSC tutorial code
    """

    Nt = len(data)
    dt = 1.0 / fs
    idx = np.argwhere(psd == 0.0)
    psd[idx] = 1e300

    # whitening: transform to freq domain, divide by asd, then transform back,
    # taking care to get normalization right.
    hf = np.fft.rfft(data)
    white_hf = hf / (np.sqrt(psd / dt / 2.))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


def gen_bbh(fs, T_obs, psds, tmp_bank, f_low, isnr=1.0, dets=['H1']):
    """
    generates a BBH timedomain signal
    """
    N = T_obs * fs  # the total number of time samples
    dt = 1 / fs  # the sampling time (sec)
    amplitude_order = 0
    phase_order = 7
    approximant = lalsimulation.IMRPhenomD
    ndet = len(dets)  # number of detectors

    # define  distribution params
    # must convert this to PyCBC-like template bank!
    #m_min = 5.0  # rest frame component masses
    #M_max = 100.0  # rest frame total mass
    #log_m_max = np.log(M_max - m_min)
    dist = 1e6 * lal.PC_SI  # put it as 1 MPc

    #flag = False
    #while not flag:
    #    m12 = np.exp(np.log(m_min) + np.random.uniform(0, 1, 2) * (log_m_max - np.log(m_min)))
        # should replace if statement here with mass > mismatch
    #    flag = True if (np.sum(m12) < M_max) and (np.all(m12 > m_min)) and (m12[0] >= m12[1]) else False

    # choose a random template from bank
    template = list(np.random.choice(np.array(tmp_bank)))
    m12 = [template[0],template[1]]
    eta = template[2]
    mc = template[3]
    print '{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(), m12[0], m12[1], mc)

    # generate iota
    iota = np.arccos(-1.0 + 2.0 * np.random.rand())
    print '{}: selected bbh cos(inclination) = {}'.format(time.asctime(), np.cos(iota))

    # generate polarisation angle
    psi = 2.0 * np.pi * np.random.rand()
    print '{}: selected bbh polarisation = {}'.format(time.asctime(), psi)

    # make waveform
    # loop until we have a long enough waveform - slowly reduce flow if needed
    flag = False
    while not flag:
        hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
            m12[0] * lal.MSUN_SI, m12[1] * lal.MSUN_SI,
            0, 0, 0, 0, 0, 0,
            dist,
            iota, 0, 0,
            0, 0,
            1 / fs,
            f_low, f_low,
            lal.CreateDict(),
            approximant)
        flag = True if hp.data.length > 2 * N else False
        f_low -= 1  # decrease by 1 Hz each time
    hp = hp.data.data
    hc = hc.data.data

    # pick sky position - uniform on the 2-sphere
    ra = 2.0 * np.pi * np.random.rand()
    dec = np.arcsin(-1.0 + 2.0 * np.random.rand())
    print '{}: selected bbh sky position = {},{}'.format(time.asctime(), ra, dec)

    # pick new random max amplitude sample location - within mid 20% region
    # and slide waveform to that location
    idx = int(np.random.randint(int(4.0 * N / 10.0), int(6.0 * N / 10.0), 1)[0])
    print '{}: selected bbh peak amplitude time = {}'.format(time.asctime(), dt * idx)

    # define point of max amplitude
    ref_idx = int(np.argmax(hp ** 2 + hc ** 2))

    # loop over detectors
    ts = np.zeros((ndet, N))
    intsnr = []
    j = 0
    for det, psd in zip(dets, psds):

        # make signal - apply antenna and shifts
        ht_temp = make_bbh(hp, hc, fs, ra, dec, psi, det)

        # place signal into timeseries - including shift
        x_temp = np.zeros(N)
        temp = ht_temp[int(ref_idx - idx):]
        if len(temp) < N:
            x_temp[:len(temp)] = temp
        else:
            x_temp = temp[:N]

            # compute SNR of pre-whitened data
        intsnr.append(get_snr(x_temp, T_obs, fs, psd.data.data))

        # don't whiten the data
        ts[j, :] = x_temp

        j += 1

    hpc_temp = np.zeros((2, N))
    temp_hp = hp[int(ref_idx - idx):]
    temp_hc = hc[int(ref_idx - idx):]
    if len(temp_hp) < N:
        hpc_temp[0, :len(temp)] = temp_hp
        hpc_temp[1, :len(temp)] = temp_hc
    else:
        hpc_temp[0, :] = temp_hp[:N]
        hpc_temp[1, :] = temp_hc[:N]

    # normalise the waveform using either integrated or peak SNR
    intsnr = np.array(intsnr)
    scale = isnr / np.sqrt(np.sum(intsnr ** 2))
    ts = scale * ts
    intsnr *= scale
    SNR = np.sqrt(np.sum(intsnr ** 2))
    print '{}: computed the network SNR = {}'.format(time.asctime(), SNR)

    # store params
    par = bbhparams(mc, m12[0], m12[1], ra, dec, np.cos(iota), psi, idx * dt, intsnr, SNR)

    return ts, par, hpc_temp


def make_bbh(hp, hc, fs, ra, dec, psi, det):
    """
    turns hplus and hcross into a detector output
    applies antenna response and
    and applies correct time delays to each detector
    """

    # make basic time vector
    tvec = np.arange(len(hp)) / float(fs)

    # compute antenna response and apply
    Fp, Fc, _, _ = antenna.response(0.0, ra, dec, 0, psi, 'radians', det)
    ht = hp * Fp + hc * Fc  # overwrite the timeseries vector to reuse it

    # compute time delays relative to Earth centre
    frDetector = lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location, ra, dec, 0.0)
    print '{}: computed {} Earth centre time delay = {}'.format(time.asctime(), det, tdelay)
    # interpolate to get time shifted signal
    tck = interpolate.splrep(tvec, ht, s=0)
    tnew = tvec + tdelay
    new_ht = interpolate.splev(tnew, tck, der=0, ext=1)

    return new_ht


def gen_ts(fs, T_obs, tmp_bank, f_low, isnr=1.0, dets=['H1']):
    """
    generates a randomly chosen timeseries from one of the available classes
    """
    N = T_obs * fs  # the total number of time samples
    dt = 1 / fs  # the sampling time (sec)
    ndet = len(dets)  # the number of detectors

    # make psds and noise realisations
    psds = [gen_psd(fs, T_obs, op='AdvDesign', det=d) for d in dets]
    noise = np.array([gen_noise(fs, T_obs, psd.data.data) for psd in psds]).reshape(ndet, -1)

    print '{}: making a BBH + noise instance'.format(time.asctime())
    sig, par, hpc = gen_bbh(fs, T_obs, psds, tmp_bank, f_low, isnr, dets)

    return sig, noise, par, hpc


def main():

    # get the command line args
    args = parser()
    np.random.seed(args.seed)

    # redefine things for conciseness
    Tobs = args.Tobs  # observation time
    fs = args.fsample  # sampling frequency
    dets = args.detectors  # detectors
    isnr = args.isnr  # integrated SNR
    ndet = len(dets)  # number of detectors
    N = Tobs * fs  # the total number of time samples
    n = N // 2 + 1  # the number of frequency bins
    tmp_bank = args.temp_bank # template bank file
    f_low = args.cutoff_freq # cutoff frequency used in template generation

    # read-in template bank
    tmp_bank = EventTable.read(tmp_bank,
    format='ligolw.sngl_inspiral', columns=['mass1','mass2','eta','mchirp'])

    psds = [gen_psd(fs, Tobs, op='AdvDesign', det=d) for d in args.detectors]
    wpsds = (2.0 / fs) * np.ones((ndet, n))  # define effective PSD for whited data

    # generate template bank

    hpcm = np.zeros(len(tmp_bank), dtype=np.ndarray)
    twhpcm = np.zeros(len(tmp_bank), dtype=np.ndarray)
    ttsm = np.zeros(len(tmp_bank), dtype=np.ndarray)
    tparm = np.zeros(len(tmp_bank), dtype=np.ndarray)


    for i in xrange(len(tmp_bank)):
        tsig, tnoise, tpar, hpc = gen_ts(fs, Tobs, tmp_bank, f_low, isnr, dets)
        whpc = np.array([whiten_data(h, psds[0].data.data, fs) for h in hpc]).reshape(2, -1)
        hpcm[i] = hpc
        twhpcm[i] = whpc
        ttsm[i] = tsig
        tparm[i] = tpar

    maxtsSNRm = np.zeros(args.Nsig)  # store maximised (over time) measured SNR
    nmaxtsSNRm = np.zeros(args.Nsig)  # store maximised (over time) noise only measured SNR
    wmaxtsSNRm = np.zeros(args.Nsig)  # store maximised (over time) measured SNR and whitened data
    wnmaxtsSNRm = np.zeros(args.Nsig)  # store maximised (over time) noise only measured SNR and whitenee data
    temp_wmaxtsSNRm = np.zeros(args.Nsig)
    temp_wnmaxtsSNRm = np.zeros(args.Nsig)
    sigparm = np.zeros(args.Nsig, dtype = np.ndarray)
    best_temp_parm = np.zeros(args.Nsig, dtype = np.ndarray)

    for i in xrange(args.Nsig):

        # generate unwhitened time-series
        sig, noise, par, hpc = gen_ts(fs, Tobs, tmp_bank, f_low, isnr, dets)
        data = sig + noise
        sigparm[i] = par

        # whiten data
        wdata = np.array([whiten_data(s, psd.data.data, fs) for s, psd in zip(data, psds)]).reshape(ndet, -1)
        print '{}: witened data variance -> {}'.format(time.asctime(), np.std(wdata[0, int(0.1 * N):int(0.9 * N)]))

        # whiten signal and template (ndet signals and 2 templates for +,x)
        wsig = np.array([whiten_data(s, psd.data.data, fs) for s, psd in zip(sig, psds)]).reshape(ndet, -1)
        whpc = np.array([whiten_data(h, psds[0].data.data, fs) for h in hpc]).reshape(2, -1)
        # TODO: replace psds[0] with correct iterable version for more than one det

        # whiten noise
        wnoise = np.array([whiten_data(s, psd.data.data, fs) for s, psd in zip(noise, psds)]).reshape(ndet, -1)
        print '{}: whitened noise variance -> {}'.format(time.asctime(), np.std(wnoise[0, int(0.1 * N):int(0.9 * N)]))

        # compute measured SNR as a function of time on whitened data
        # using LOSC convolution method
        # wtsSNR = np.array([snr_ts(d, whpc[0], whpc[1], Tobs, fs, p) for d, p in zip(wdata, wpsds)])
        # wmaxtsSNR = np.max(wtsSNR, axis=1)
        # wmaxtsSNRm[i] = np.sqrt(np.sum(wmaxtsSNR ** 2))
        # print '{}: maximised single detector SNR = {}'.format(time.asctime(), wmaxtsSNRm[i])

        # compute measured SNR *on noise only* as a function of time
        # using LOSC convolution method
        # wntsSNR = np.array([snr_ts(n, whpc[0], whpc[1], Tobs, fs, p) for n, p in zip(wnoise, wpsds)])
        # wnmaxtsSNR = np.max(wntsSNR, axis=1)
        # wnmaxtsSNRm[i] = np.sqrt(np.sum(wnmaxtsSNR ** 2))
        # print '{}: maximised noise only single detector SNR = {}'.format(time.asctime(), wnmaxtsSNRm[i])
        # TODO: add cacpacity for 2 detectors
        # set best max SNR

        best_temp_wmaxtsSNR = 0
        for n,twhpc in enumerate(twhpcm):
            temp_wtsSNR = np.array([snr_ts(d,twhpc[0],twhpc[1],Tobs,fs,p) for d,p in zip(wdata,wpsds)])
            temp_wmaxtsSNR = np.max(temp_wtsSNR[-2048:])
            # save max SNR if greater than previous
            if temp_wmaxtsSNR > best_temp_wmaxtsSNR:
                best_temp_wmaxtsSNR = np.copy(temp_wmaxtsSNR)
                nt = n
        temp_wmaxtsSNRm[i] = np.sqrt(np.sum(best_temp_wmaxtsSNR**2.0))
        best_temp_parm[i] = tparm[nt]
        #d_eff = bwtssigma / bwmaxtsSNR
        print '{}: template maximised single detector SNR = {}'.format(time.asctime(), temp_wmaxtsSNRm[i])
        print '{}: template masses = {},{} (chirp mass = {})'.format(time.asctime(), tparm[nt].m1, tparm[nt].m2, tparm[nt].mc)
        # compute measured SNR *on noise only* as a function of time on
        # whitened data using LOSC convolution method
        # set best max SNR
        best_temp_wnmaxtsSNR = 0
        for twhpc in twhpcm:
            temp_wntsSNR = np.array([snr_ts(d,twhpc[0],twhpc[1],Tobs,fs,p) for d,p in zip(wnoise,wpsds)])
            temp_wnmaxtsSNR = np.max(temp_wntsSNR)
            # save max SNR is great than previous
            if temp_wnmaxtsSNR > best_temp_wnmaxtsSNR:
                best_temp_wnmaxtsSNR = np.copy(temp_wnmaxtsSNR)
        temp_wnmaxtsSNRm[i] = np.sqrt(np.sum(best_temp_wnmaxtsSNR**2.0))
        print '{}: template maximised noise only single detector SNR = {}'.format(time.asctime(), temp_wnmaxtsSNRm[i])
        print '{}: completed {}/{} time series'.format(time.asctime(),i+1,args.Nsig)
        print ('{}: current mean iSNR of {} signals = {}'.format(time.asctime(), i+1, np.mean(wmaxtsSNRm[:i])))
        print ('{}: current template mean iSNR of {} signals = {}'.format(time.asctime(), i+1, np.mean(temp_wmaxtsSNRm[:i])))

    # make distribution plots
    nbins = int(np.sqrt(args.Nsig))
    temp = np.linspace(0, isnr + 5, 1000)
    plt.figure()
    plt.hist(wmaxtsSNRm, nbins, normed=True, alpha=0.5)
    plt.hist(wnmaxtsSNRm, nbins, normed=True, alpha=0.5)
    plt.hist(temp_wmaxtsSNRm, nbins, normed=True, alpha=0.5)
    plt.hist(temp_wnmaxtsSNRm, nbins, normed=True, alpha=0.5)
    plt.plot(temp, norm.pdf(temp, loc=isnr), 'k')
    plt.plot(temp, chi.pdf(temp, 2), 'k')
    plt.xlim([0, np.max(temp)])
    plt.savefig('./verify_whitened.png')


    # if args.Nsig == 1:
    #     t = np.linspace(0,1,8192)
    #     print len(sig[0])
    #     plt.figure()
    #     plt.plot(t, sig[0], label = 'signal')
    #     plt.plot(t, hpcm[nt][0]/d_eff, label = 'template p')
    #     plt.plot(t, hpcm[nt][1]/d_eff, label = 'template c')
    #     plt.legend()
    #     plt.savefig('./time_series.png')
    #
    #     wtemp = (whpcm[nt][0] + whpcm[nt][1]*1.j)
    #     wtemp_phase = np.real(wtemp*np.exp(1j*bwtsphase))
    #
    #     plt.figure()
    #     plt.plot(t,wsig[0], label = 'whitened signal')
    #     plt.plot(t, wtemp_phase/d_eff, label = 'whitened template')
    #     plt.legend()
    #     plt.savefig('./whitened_time_series.png')
    #
    # # make distribution of masses
    # cmap = plt.get_cmap('jet')
    # plt.figure()
    #
    # for sp,tp,c in zip(sigparm,mtparm,range(len(sigparm))):
    #     plt.plot(sp.m1, sp.m2, c='red', marker = '.')
    #     plt.plot(tp.m1, tp.m2, c='green', marker = '.')
    # plt.savefig('./masses.png')
    #
    # mcm = [tp.mc for tp in tparm]
    #
    # plt.figure()
    # nbins = np.sqrt(len(mcm))
    # print len(mcm)
    # plt.hist(mcm, nbins, normed=True)
    # plt.savefig('./mass_hist.png')


    # calculate elements of confusion matrix for ROC curve
    FPR = []
    TPR = []
    FNR = []
    TNR = []
    # number of steps
    steps = 100
    thresholds = np.linspace(temp_wnmaxtsSNRm.min(), temp_wmaxtsSNRm.max(), num=steps)
    Nsignals = len(temp_wmaxtsSNRm)
    Nnoise = len(temp_wnmaxtsSNRm)

    for i in thresholds:
        TP = len(temp_wmaxtsSNRm[temp_wmaxtsSNRm > i])
        TN = len(temp_wnmaxtsSNRm[temp_wnmaxtsSNRm < i])
        FN = len(temp_wmaxtsSNRm[temp_wmaxtsSNRm < i])
        FP = len(temp_wnmaxtsSNRm[temp_wnmaxtsSNRm > i])
        FPR.append(FP)
        TPR.append(TP)
        FNR.append(FN)
        TNR.append(TN)

    # save all results together
    res = ROCresults(TPR, FNR, FPR, TNR, thresholds, Nsignals, Nnoise, isnr)

    if not os.path.exists('./ROCdata'):
        os.makedirs('./ROCdata')

    with open('./ROCdata/ROC_test_8K_iSNR{0}.pkl'.format(isnr), 'wb') as wfp:
        pickle.dump(res, wfp)
        print ('Saved ROC results as ROC_800temps_iSNR{0}.pkl'.format(isnr))

if __name__ == '__main__':
    main()
