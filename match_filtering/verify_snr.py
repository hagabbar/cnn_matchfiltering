from __future__ import division
import cPickle
import numpy as np
from scipy import integrate, interpolate
from scipy.misc import imsave
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import lal
import lalsimulation
from pylal import antenna, cosmography
import argparse
import time
from scipy.signal import filtfilt, butter, tukey
from scipy.stats import norm, chi
import os

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

def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='data_prep.py',description='generates GW data for application of deep learning networks.')

    # arguments for reading in a data file
    parser.add_argument('-N', '--Nsig', type=int, default=3000, help='the number of example signals')
    parser.add_argument('-f', '--fsample', type=int, default=8192, help='the sampling frequency (Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=1, help='the observation duration (sec)')
    parser.add_argument('-s', '--isnr', type=float, default=8, help='the signal integrated SNR')   
    parser.add_argument('-I', '--detectors', type=str, nargs='+',default=['H1','L1'], help='the detectors to use')   
    parser.add_argument('-b', '--basename', type=str,default='test', help='output file path and basename')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')

    return parser.parse_args()

def gen_noise(fs,T_obs,psd):
    """
    Generates noise from a psd
    """

    N = T_obs * fs          # the total number of time samples
    Nf = N // 2 + 1
    dt = 1 / fs             # the sampling time (sec)
    df = 1 / T_obs

    amp = np.sqrt(0.25*T_obs*psd)
    idx = np.argwhere(psd==0.0)
    amp[idx] = 0.0
    idx = np.argwhere(psd==1e300)
    amp[idx] = 0.0
    re = amp*np.random.normal(0,1,Nf)
    im = amp*np.random.normal(0,1,Nf)
    re[0] = 0.0
    im[0] = 0.0
    x = N*np.fft.irfft(re + 1j*im)*df

    return x

def gen_psd(fs,T_obs,op='AdvDesign',det='H1'):
    """
    generates noise for a variety of different detectors
    """
    N = T_obs * fs          # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    df = 1 / T_obs          # the frequency resolution
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df,lal.HertzUnit, N // 2 + 1)    

    if det=='H1' or det=='L1':
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

def get_snr(data,T_obs,fs,psd):
    """
    computes the snr of a whitened signal in unit variance time domain noise
    """

    N = T_obs*fs
    df = 1.0/T_obs
    dt = 1.0/fs

    win = tukey(N,alpha=1.0/8.0)
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300    

    xf = np.fft.rfft(data*win)*dt
    SNRsq = 4.0*np.sum((np.abs(xf)**2)/psd)*df
    return np.sqrt(SNRsq)

def inner(a,b,T_obs,fs,psd):
    """
    Computes the noise weighted inner product in the frequency domain
    Follows Babak et al Eq. 2
    """
    N = T_obs*fs
    df = 1.0/T_obs
    dt = 1.0/fs

    win = tukey(N,alpha=1.0/8.0)
    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300

    af = np.fft.rfft(a*win)*dt
    bf = np.fft.rfft(b*win)*dt  
    temp = 4.0*np.real(np.sum((np.conj(af)*bf)/psd))*df
    return temp

def meas_snr(data,template_p,template_c,Tobs,fs,psd):
    """
    Computes the measured SNR for a given template and dataset 
    Follows Babak et al Eq. 9
    """

    a = inner(data,template_p,Tobs,fs,psd)
    b = inner(data,template_c,Tobs,fs,psd)
    c = inner(template_p,template_p,Tobs,fs,psd)
    return np.sqrt((a*a + b*b)/c)

def snr_ts(data,template_p,template_c,Tobs,fs,psd):
    """
    Computes the SNR for each time step
    Based on the LOSC tutorial code 
    """

    N = Tobs*fs
    df = 1.0/Tobs
    dt = 1.0/fs

    temp = template_p + template_c*1.j
    dwindow = tukey(temp.size, alpha=1./8)
    #dwindow = np.ones(temp.size)

    idx = np.argwhere(psd==0.0)
    psd[idx] = 1e300

    # Take the Fourier Transform (FFT) of the data and the template (with dwindow)
    data_fft = np.fft.fft(data*dwindow)*dt
    template_fft = np.fft.fft(temp*dwindow)*dt
    freqs = np.fft.fftfreq(N,dt)
    oldfreqs = df*np.arange(N//2 + 1)

    intpsd = np.interp(np.abs(freqs),oldfreqs,psd)
    idx = np.argwhere(intpsd==0.0)
    intpsd[idx] = 1e300
    idx = np.argwhere(np.isnan(intpsd))
    intpsd[idx] = 1e300

    # -- Calculate the matched filter output in the time domain:
    # Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
    # Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
    # so the result will be plotted as a function of time off-set between the template and the data:
    optimal = data_fft * template_fft.conjugate() / intpsd
    optimal_time = 2*np.fft.ifft(optimal)*fs

    # -- Normalize the matched filter output:
    # Normalize the matched filter output so that we expect a value of 1 at times of just noise.
    # Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
    sigmasq = 1*(template_fft * template_fft.conjugate() / intpsd).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time/sigma

    return abs(SNR_complex)

def whiten_data(data, psd, fs):
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

def gen_bbh(fs,T_obs,psds,isnr=1.0,dets=['H1']):
    """
    generates a BBH timedomain signal
    """
    N = T_obs * fs      # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    f_low = 12.0            # lowest frequency of waveform (Hz)
    amplitude_order = 0
    phase_order = 7
    approximant = lalsimulation.IMRPhenomD
    ndet = len(dets)    # number of detectors

    # define distribution params
    m_min = 5.0         # rest frame component masses
    M_max = 100.0       # rest frame total mass
    log_m_max = np.log(M_max - m_min)
    dist = 1e6*lal.PC_SI  # put it as 1 MPc

    flag = False
    while not flag:
        m12 = np.exp(np.log(m_min) + np.random.uniform(0,1,2)*(log_m_max-np.log(m_min)))
        flag = True if (np.sum(m12)<M_max) and (np.all(m12>m_min)) and (m12[0]>=m12[1]) else False
    eta = m12[0]*m12[1]/(m12[0]+m12[1])**2
    mc = np.sum(m12)*eta**(3.0/5.0)
    print '{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(),m12[0],m12[1],mc)

    # generate iota
    iota = np.arccos(-1.0 + 2.0*np.random.rand())
    print '{}: selected bbh cos(inclination) = {}'.format(time.asctime(),np.cos(iota))    

    # generate polarisation angle 
    psi = 2.0*np.pi*np.random.rand()
    print '{}: selected bbh polarisation = {}'.format(time.asctime(),psi)   

    # make waveform
    # loop until we have a long enough waveform - slowly reduce flow is needed
    flag = False
    while not flag:
        hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
                    m12[0] * lal.MSUN_SI, m12[1] * lal.MSUN_SI,
                    0, 0, 0, 0, 0, 0,
                    dist,
                    iota, 0, 0,
                    0, 0,
                    1 / fs, 
                    f_low,f_low,
                    lal.CreateDict(),
                    approximant)
        flag = True if hp.data.length>2*N else False
        f_low -= 1       # decrease by 1 Hz each time
    hp = hp.data.data
    hc = hc.data.data

    # pick sky position - uniform on the 2-sphere
    ra = 2.0*np.pi*np.random.rand()   
    dec = np.arcsin(-1.0 + 2.0*np.random.rand())
    print '{}: selected bbh sky position = {},{}'.format(time.asctime(),ra,dec)

    # pick new random max amplitude sample location - within mid 20% region
    # and slide waveform to that location
    idx = int(np.random.randint(int(4.0*N/10.0),int(6.0*N/10.0),1)[0])
    print '{}: selected bbh peak amplitude time = {}'.format(time.asctime(),dt*idx)

    # define point of max amplitude
    ref_idx = int(np.argmax(hp**2 + hc**2))

    # loop over detectors
    ts = np.zeros((ndet,N))
    intsnr = []
    j = 0
    for det,psd in zip(dets,psds):

        # make signal - apply antenna and shifts
        ht_temp = make_bbh(hp,hc,fs,ra,dec,psi,det)
    
        # place signal into timeseries - including shift
        x_temp = np.zeros(N)
        temp = ht_temp[int(ref_idx-idx):]
        if len(temp)<N:
            x_temp[:len(temp)] = temp
        else:
            x_temp = temp[:N]        

        # compute SNR of pre-whitened data
        intsnr.append(get_snr(x_temp,T_obs,fs,psd.data.data))

        # don't whiten the data
        ts[j,:] = x_temp

        j += 1

    hpc_temp = np.zeros((2,N))
    temp_hp = hp[int(ref_idx-idx):]
    temp_hc = hc[int(ref_idx-idx):]
    if len(temp_hp)<N:
        hpc_temp[0,:len(temp)] = temp_hp
        hpc_temp[1,:len(temp)] = temp_hc
    else:
        hpc_temp[0,:] = temp_hp[:N]
        hpc_temp[1,:] = temp_hc[:N]

    # normalise the waveform using either integrated or peak SNR
    intsnr = np.array(intsnr)
    scale = isnr/np.sqrt(np.sum(intsnr**2))
    ts = scale*ts
    intsnr *= scale
    SNR = np.sqrt(np.sum(intsnr**2))
    print '{}: computed the network SNR = {}'.format(time.asctime(),SNR)

    # store params
    par = bbhparams(mc,m12[0],m12[1],ra,dec,np.cos(iota),psi,idx*dt,intsnr,SNR)

    return ts, par, hpc_temp

def make_bbh(hp,hc,fs,ra,dec,psi,det):
    """
    turns hplus and hcross into a detector output
    applies antenna response and
    and applies correct time delays to each detector
    """

    # make basic time vector
    tvec = np.arange(len(hp))/float(fs)

    # compute antenna response and apply
    Fp,Fc,_,_ = antenna.response( 0.0, ra, dec, 0, psi, 'radians', det )
    ht = hp*Fp + hc*Fc     # overwrite the timeseries vector to reuse it

    # compute time delays relative to Earth centre
    frDetector =  lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location,ra,dec,0.0)
    print '{}: computed {} Earth centre time delay = {}'.format(time.asctime(),det,tdelay)
    # interpolate to get time shifted signal
    tck = interpolate.splrep(tvec, ht, s=0)
    tnew = tvec + tdelay
    new_ht = interpolate.splev(tnew, tck, der=0,ext=1)

    return new_ht

def gen_ts(fs,T_obs,isnr=1.0,dets=['H1']):
    """ 
    generates a randomly chosen timeseries from one of the available classes
    """
    N = T_obs * fs          # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)    
    ndet = len(dets)            # the number of detectors
 
    # make psds and noise realisations   
    psds = [gen_psd(fs,T_obs,op='AdvDesign',det=d) for d in dets]
    noise = np.array([gen_noise(fs,T_obs,psd.data.data) for psd in psds]).reshape(ndet,-1) 

    print '{}: making a BBH + noise instance'.format(time.asctime())
    sig, par, hpc = gen_bbh(fs,T_obs,psds,isnr,dets)
    
    return sig, noise, par, hpc

# the main part of the code
def main():
    """
    The main code - generates the training, validation and test samples
    """

    # get the command line args
    args = parser()
    np.random.seed(args.seed)
    
    # redefine things for conciseness
    Tobs = args.Tobs            # observation time
    fs = args.fsample           # sampling frequency
    dets = args.detectors       # detectors
    isnr = args.isnr            # integrated SNR
    ndet = len(dets)            # number of detectors
    N = Tobs*fs                 # the total number of time samples
    n = N // 2 + 1              # the number of frequency bins
    
    # make the psds
    psds = [gen_psd(fs,Tobs,op='AdvDesign',det=d) for d in args.detectors]
    wpsds = (2.0/fs)*np.ones((ndet,n))          # define effective PSD for whited data

    # loop over signals
    peakSNRvec = []                     # store individual detector peak SNRs
    intSNRvec = []                      # store indiviual detector integrated SNRs
    optSNRm = np.zeros(args.Nsig)       # store optimal SNR
    testSNRm = np.zeros(args.Nsig)      # store alternative optimal SNR 
    woptSNRm = np.zeros(args.Nsig)      # store optimal SNR for whitened signal
    wtestSNRm = np.zeros(args.Nsig)     # store alternative optimal SNR for whitened data
    filtSNRm = np.zeros(args.Nsig)      # store exact filter measured SNR
    nfiltSNRm = np.zeros(args.Nsig)     # store exact filter noise only measured SNR
    wfiltSNRm = np.zeros(args.Nsig)     # store exact filter measured SNR for whitened data
    wnfiltSNRm = np.zeros(args.Nsig)    # store exact filter noise only measured SNR and whitened data
    maxtsSNRm = np.zeros(args.Nsig)     # store maximised (over time) measured SNR
    nmaxtsSNRm = np.zeros(args.Nsig)    # store maximised (over time) noise only measured SNR
    wmaxtsSNRm = np.zeros(args.Nsig)    # store maximised (over time) measured SNR and whitened data
    wnmaxtsSNRm = np.zeros(args.Nsig)   # store maximised (over time) noise only measured SNR and whitenee data 
    print '{}: starting to generate data'.format(time.asctime())
    for i in xrange(args.Nsig):
    
        # generate unwhitened time-series
        sig,noise,par,hpc = gen_ts(fs,Tobs,isnr,dets)
        data = sig + noise

        # whiten data
        wdata = np.array([whiten_data(s,psd.data.data,fs) for s,psd in zip(data,psds)]).reshape(ndet,-1)
        print '{}: Whitened data variance -> {}'.format(time.asctime(),np.std(wdata[0,int(0.1*N):int(0.9*N)]))

        # whiten signal and template (ndet signals and 2 templates for +,x)
        wsig = np.array([whiten_data(s,psd.data.data,fs) for s,psd in zip(sig,psds)]).reshape(ndet,-1)
        whpc = np.array([whiten_data(h,psd.data.data,fs) for h,psd in zip(hpc,psds)]).reshape(2,-1)
        peakSNRvec.append(np.max(np.abs(wsig),axis=1))
        
        # whiten noise
        wnoise = np.array([whiten_data(s,psd.data.data,fs) for s,psd in zip(noise,psds)]).reshape(ndet,-1)
        print '{}: Whitened noise variance -> {}'.format(time.asctime(),np.std(wnoise[0,int(0.1*N):int(0.9*N)]))

        # compute optimal SNR in 2 different ways
        optSNR = np.array([get_snr(s,Tobs,fs,p.data.data) for s,p in zip(sig,psds)])
        testSNRsq = np.array([inner(s,s,Tobs,fs,p.data.data) for s,p in zip(sig,psds)])
        optSNRm[i] = np.sqrt(np.sum(optSNR**2))
        testSNRm[i] = np.sqrt(np.sum(testSNRsq))        
        intSNRvec.append(optSNR)
        print '{}: optimal multidetector SNR = {}'.format(time.asctime(),optSNRm[i])
        print '{}: optimal multidetector SNR (test) = {}'.format(time.asctime(),testSNRm[i])

        # compute optimal SNR for whitened signal
        woptSNR = np.array([get_snr(s,Tobs,fs,p) for s,p in zip(wsig,wpsds)])
        wtestSNRsq = np.array([inner(s,s,Tobs,fs,p) for s,p in zip(wsig,wpsds)])
        woptSNRm[i] = np.sqrt(np.sum(woptSNR**2))
        wtestSNRm[i] = np.sqrt(np.sum(wtestSNRsq))
        print '{}: optimal multidetector SNR (whited signal) = {}'.format(time.asctime(),woptSNRm[i])
        print '{}: optimal multidetector SNR (whitened data test) = {}'.format(time.asctime(),wtestSNRm[i])
        
        # compute measured SNR using the exact template
        filtSNR = np.array([meas_snr(d,s,np.zeros(N),Tobs,fs,p.data.data) for d,s,p in zip(data,sig,psds)])
        filtSNRm[i] = np.sqrt(np.sum(filtSNR**2))
        print '{}: exact template multidetector SNR = {}'.format(time.asctime(),filtSNRm[i])        

        # compute measured SNR on whitened data using the exact template
        wfiltSNR = np.array([meas_snr(d,s,np.zeros(N),Tobs,fs,p) for d,s,p in zip(wdata,wsig,wpsds)])
        wfiltSNRm[i] = np.sqrt(np.sum(wfiltSNR**2))
        print '{}: exact template multidetector SNR (whitened data) = {}'.format(time.asctime(),wfiltSNRm[i])

        # compute measured SNR *on noise only* using the exact template
        nfiltSNR = np.array([meas_snr(n,s,np.zeros(N),Tobs,fs,p.data.data) for n,s,p in zip(noise,sig,psds)])
        nfiltSNRm[i] = np.sqrt(np.sum(nfiltSNR**2))
        print '{}: exact template noise only multidetector SNR = {}'.format(time.asctime(),nfiltSNRm[i])       

        # compute measured SNR *on noise only* whitened data using the exact template
        wnfiltSNR = np.array([meas_snr(n,s,np.zeros(N),Tobs,fs,p) for n,s,p in zip(wnoise,wsig,wpsds)])
        wnfiltSNRm[i] = np.sqrt(np.sum(wnfiltSNR**2))
        print '{}: exact template noise only multidetector SNR (whitened data) = {}'.format(time.asctime(),wnfiltSNRm[i])

        # compute measured SNR as a function of time
        # using LOSC convolution method
        tsSNR = np.array([snr_ts(d,hpc[0],hpc[1],Tobs,fs,p.data.data) for d,p in zip(data,psds)])
        maxtsSNR = np.max(tsSNR,axis=1)
        maxtsSNRm[i] = np.sqrt(np.sum(maxtsSNR**2))
        print '{}: maximised multidetector SNR = {}'.format(time.asctime(),maxtsSNRm[i])

        # compute measured SNR *on noise only* as a function of time
        # using LOSC convolution method
        ntsSNR = np.array([snr_ts(n,hpc[0],hpc[1],Tobs,fs,p.data.data) for n,p in zip(noise,psds)])
        nmaxtsSNR = np.max(ntsSNR,axis=1)
        nmaxtsSNRm[i] = np.sqrt(np.sum(nmaxtsSNR**2))
        print '{}: maximised noise only multidetector SNR = {}'.format(time.asctime(),nmaxtsSNRm[i])

        # compute measured SNR as a function of time on whitened data
        # using LOSC convolution method
        wtsSNR = np.array([snr_ts(d,whpc[0],whpc[1],Tobs,fs,p) for d,p in zip(wdata,wpsds)])
        wmaxtsSNR = np.max(wtsSNR,axis=1)
        wmaxtsSNRm[i] = np.sqrt(np.sum(wmaxtsSNR**2))
        print '{}: maximised multidetector SNR = {}'.format(time.asctime(),wmaxtsSNRm[i])

        # compute measured SNR *on noise only* as a function of time on
        # whitened data using LOSC convolution method
        wntsSNR = np.array([snr_ts(n,whpc[0],whpc[1],Tobs,fs,p) for n,p in zip(wnoise,wpsds)])
        wnmaxtsSNR = np.max(wntsSNR,axis=1)
        wnmaxtsSNRm[i] = np.sqrt(np.sum(wnmaxtsSNR**2))
        print '{}: maximised noise only multidetector SNR = {}'.format(time.asctime(),wnmaxtsSNRm[i])

    # make distribution plots
    nbins = int(np.sqrt(args.Nsig))
    temp = np.linspace(0,isnr+5,1000)
    plt.figure()
    plt.hist(filtSNRm,nbins,normed=True,alpha=0.5)
    plt.hist(maxtsSNRm,nbins,normed=True,alpha=0.5)
    plt.hist(nfiltSNRm,nbins,normed=True,alpha=0.5)
    plt.hist(nmaxtsSNRm,nbins,normed=True,alpha=0.5)
    plt.plot(temp,norm.pdf(temp,loc=isnr),'k')
    plt.plot(temp,chi.pdf(temp,2),'k')
    plt.xlim([0,np.max(temp)]) 
    plt.savefig('./verify.png')
    
    # make distribution plots
    plt.figure()
    plt.hist(wfiltSNRm,nbins,normed=True,alpha=0.5)
    plt.hist(wmaxtsSNRm,nbins,normed=True,alpha=0.5)
    plt.hist(wnfiltSNRm,nbins,normed=True,alpha=0.5)
    plt.hist(wnmaxtsSNRm,nbins,normed=True,alpha=0.5)
    plt.plot(temp,norm.pdf(temp,loc=isnr),'k')
    plt.plot(temp,chi.pdf(temp,2),'k')
    plt.xlim([0,np.max(temp)])
    plt.savefig('./verify_whitened.png')

    # make peak vs int snr plots
    peakSNRvec = np.array(peakSNRvec).flatten()
    intSNRvec = np.array(intSNRvec).flatten()
    plt.figure()
    plt.plot(peakSNRvec,intSNRvec,'.')
    plt.xlim([0,1.2*np.max(peakSNRvec)])
    plt.ylim([0,1.2*np.max(intSNRvec)])
    plt.savefig('./peakvsint.png')
    plt.figure()
    plt.hist(intSNRvec/peakSNRvec,nbins,normed=True,alpha=0.5)
    plt.savefig('./peakintratio.png')

if __name__ == "__main__":
    exit(main())
