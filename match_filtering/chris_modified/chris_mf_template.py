from __future__ import division
import cPickle as pickle
import numpy as np
from gwpy.table import EventTable
from scipy import integrate, interpolate
from scipy.misc import imsave
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import lal
from lal import MSUN_SI, C_SI, G_SI
import lalsimulation
from pylal import antenna, cosmography
import argparse
import time
from scipy.signal import filtfilt, butter #, tukey
from scipy.stats import norm, chi
from scipy.optimize import brentq
from numpy.fft import fft, ifft, rfft, irfft, fftfreq
#from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq
import os
safe = 2

class bbhparams:
    def __init__(self,mc,M,eta,m1,m2,ra,dec,iota,psi,idx,fmin,snr,SNR):
        self.mc = mc
        self.M = M
        self.eta = eta
        self.m1 = m1
        self.m2 = m2
        self.ra = ra
        self.dec = dec
        self.iota = iota
        self.psi = psi
        self.idx = idx
        self.fmin = fmin
        self.snr = snr
        self.SNR = SNR

def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='data_prep.py',description='generates GW data for application of deep learning networks.')

    # arguments for reading in a data file
    #parser.add_argument('-N', '--Nsig', type=int, default=3000, help='the number of example signals')
    #parser.add_argument('-n', '--Ntemp', type=int, default=3000, help='the number of templates per signal')
    parser.add_argument('-f', '--fsample', type=int, default=8192, help='the sampling frequency (Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=1, help='the observation duration (sec)')
    parser.add_argument('-s', '--isnr', type=float, default=7, help='the signal integrated SNR')   
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')
    parser.add_argument('-tb', '--temp-bank', type=str, help='template bank .xml file')
    parser.add_argument('-d', '--dataset', type=str, help='test set')
    parser.add_argument('-b', '--basename', type=str, default='test', help='output file path and basename.')

    return parser.parse_args()

def tukey(M,alpha=0.5):
    """
    Tukey window code copied from scipy
    """
    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
    w = np.concatenate((w1, w2, w3))

    return np.array(w[:M])

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
    re = amp*np.random.normal(0,1,Nf)
    im = amp*np.random.normal(0,1,Nf)
    re[0] = 0.0
    im[0] = 0.0
    x = N*irfft(re + 1j*im)*df

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

def get_fmin(M,eta,dt):
    """
    Compute the instantaneous frequency given a time till merger
    """
    M_SI = M*MSUN_SI

    def dtchirp(f):
        """
        The chirp time to 2nd PN order
        """
        v = ((G_SI/C_SI**3)*M_SI*np.pi*f)**(1.0/3.0)
        temp = (v**(-8.0) + ((743.0/252.0) + 11.0*eta/3.0)*v**(-6.0) -
                (32*np.pi/5.0)*v**(-5.0) + ((3058673.0/508032.0) + 5429*eta/504.0 +
                (617.0/72.0)*eta**2)*v**(-4.0))
        return (5.0/(256.0*eta))*(G_SI/C_SI**3)*M_SI*temp - dt

    # solve for the frequency between limits
    fmin = brentq(dtchirp, 1.0, 2000.0, xtol=1e-6)
    #print '{}: signal enters segment at {} Hz'.format(time.asctime(),fmin)

    return fmin

def get_snr(data,T_obs,fs,psd,fmin):
    """
    computes the snr of a signal given a PSD starting from a particular frequency index
    """

    N = T_obs*fs
    df = 1.0/T_obs
    dt = 1.0/fs
    fidx = int(fmin/df)

    win = tukey(N,alpha=1.0/8.0)
    idx = np.argwhere(psd>0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0/psd[idx]

    xf = rfft(win*data)*dt
    SNRsq = 4.0*np.sum((np.abs(xf[fidx:])**2)*invpsd[fidx:])*df
    return np.sqrt(SNRsq)

def snr_ts(data,template_p,template_c,Tobs,fs,psd,fmin,flag='td'):
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
    freqs = fftfreq(N,dt)
    oldfreqs = df*np.arange(N//2 + 1)
    intpsd = np.interp(np.abs(freqs),oldfreqs,psd)
    idx = np.argwhere(intpsd==0.0)
    intpsd[idx] = 1e300
    idx = np.argwhere(np.isnan(intpsd))
    intpsd[idx] = 1e300

    if flag=='td':
        # make complex template
        temp = template_p + template_c*1.j
        ftemp = fft(temp*win)*dt
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
    fdata = fft(data*win)*dt

    z = 4.0*ifft(fdata*np.conj(ftemp)/intpsd)*df*N
    s = 4.0*np.sum(np.abs(ftemp)**2/intpsd)*df
    return np.abs(z)/np.sqrt(s)

def whiten_data(data,duration,sample_rate,psd,flag='td'):
    """
    Takes an input timeseries and whitens it according to a psd
    """

    if flag=='td':
        # FT the input timeseries - window first
        win = tukey(duration*sample_rate,alpha=1.0/8.0)
        xf = rfft(win*data)
    else:
        xf = data

    # deal with undefined PDS bins and normalise
    idx = np.argwhere(psd>0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0/psd[idx]
    xf *= np.sqrt(2.0*invpsd/sample_rate)

    # Detrend the data: no DC component.
    xf[0] = 0.0

    if flag=='td':
        # Return to time domain.
        x = irfft(xf)
        return x
    else:
        return xf

def gen_par(template,fs,T_obs,beta=[0.8,1.0]):
    """
    Generates a random set of parameters
    """

    # define template parameters
    template = list(template)
    m12 = [template[0],template[1]]
    M = np.sum(m12)
    eta = template[2]
    mc = template[3]

    # define distribution params
    #m_min = 5.0         # rest frame component masses
    #M_max = 100.0       # rest frame total mass
    #log_m_max = np.log(M_max - m_min)

    #flag = False
    #while not flag:
    #    m12 = np.exp(np.log(m_min) + np.random.uniform(0,1,2)*(log_m_max-np.log(m_min)))
    #    flag = True if (np.sum(m12)<M_max) and (np.all(m12>m_min)) and (m12[0]>=m12[1]) else False
    #M = np.sum(m12)
    #eta = m12[0]*m12[1]/M**2
    #mc = M*eta**(3.0/5.0)
    #print '{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(),m12[0],m12[1],mc)

    # generate iota
    iota = np.arccos(-1.0 + 2.0*np.random.rand())
    #print '{}: selected bbh cos(inclination) = {}'.format(time.asctime(),np.cos(iota))

    # generate polarisation angle
    psi = 2.0*np.pi*np.random.rand()
    #print '{}: selected bbh polarisation = {}'.format(time.asctime(),psi)

    # pick sky position - uniform on the 2-sphere
    ra = 2.0*np.pi*np.random.rand()
    dec = np.arcsin(-1.0 + 2.0*np.random.rand())
    #print '{}: selected bbh sky position = {},{}'.format(time.asctime(),ra,dec)

    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    low_idx,high_idx = convert_beta(beta,fs,T_obs)
    idx = int(np.random.randint(low_idx,high_idx,1)[0])
    #print '{}: selected bbh peak amplitude time = {}'.format(time.asctime(),idx/fs)

    # the start index of the central region
    sidx = int(0.5*fs*T_obs*(safe-1.0)/safe)

    # compute SNR of pre-whitened data
    fmin = get_fmin(M,eta,int(idx-sidx)/fs)
    #print '{}: computed starting frequency = {} Hz'.format(time.asctime(),fmin)

    # store params
    par = bbhparams(mc,M,eta,m12[0],m12[1],ra,dec,np.cos(iota),psi,idx,fmin,None,None)

    return par

def convert_beta(beta,fs,T_obs):
    """
    Converts beta values (fractions defining a desired period of time in
    central output window) into indices for the full safe time window 
    """
    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    newbeta = np.array([(beta[0] + 0.5*safe - 0.5),(beta[1] + 0.5*safe - 0.5)])/safe
    low_idx = int(T_obs*fs*newbeta[0])
    high_idx = int(T_obs*fs*newbeta[1])
    
    return low_idx,high_idx

def gen_bbh(fs,T_obs,psd,isnr=1.0,det='H1',beta=[0.8,1.0],par=None):
    """
    generates a BBH timedomain signal
    """
    N = T_obs * fs      # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    f_low = 12.0            # lowest frequency of waveform (Hz)
    amplitude_order = 0
    phase_order = 7
    approximant = lalsimulation.IMRPhenomD
    dist = 1e6*lal.PC_SI  # put it as 1 MPc

    # make waveform
    # loop until we have a long enough waveform - slowly reduce flow as needed
    flag = False
    while not flag:
        hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
                    par.m1 * lal.MSUN_SI, par.m2 * lal.MSUN_SI,
                    0, 0, 0, 0, 0, 0,
                    dist,
                    par.iota, 0, 0,
                    0, 0,
                    1 / fs, 
                    f_low,f_low,
                    lal.CreateDict(),
                    approximant)
        flag = True if hp.data.length>2*N else False
        f_low -= 1       # decrease by 1 Hz each time
    hp = hp.data.data
    hc = hc.data.data

    # compute reference idx
    ref_idx = np.argmax(hp**2 + hc**2)

    # make signal - apply antenna and shifts
    ht_shift, hp_shift, hc_shift = make_bbh(hp,hc,fs,par.ra,par.dec,par.psi,det)
    
    # place signal into timeseries - including shift
    ts = np.zeros(N)
    hp = np.zeros(N)
    hc = np.zeros(N)
    ht_temp = ht_shift[int(ref_idx-par.idx):]
    hp_temp = hp_shift[int(ref_idx-par.idx):]
    hc_temp = hc_shift[int(ref_idx-par.idx):]
    if len(ht_temp)<N:
        ts[:len(ht_temp)] = ht_temp
        hp[:len(ht_temp)] = hp_temp
        hc[:len(ht_temp)] = hc_temp
    else:
        ts = ht_temp[:N]
        hp = hp_temp[:N]
        hc = hc_temp[:N]
   
    # the start index of the central region
    sidx = int(0.5*fs*T_obs*(safe-1.0)/safe)

    # apply aggressive window to cut out signal in central region
    # window is non-flat for 1/8 of desired Tobs
    # the window has dropped to 50% at the Tobs boundaries
    win = np.zeros(N)
    tempwin = tukey(int((16.0/15.0)*N/safe),alpha=1.0/8.0)
    win[int((N-tempwin.size)/2):int((N-tempwin.size)/2)+tempwin.size] = tempwin
    ts *= win
    hp *= win
    hc *= win

    # compute SNR of pre-whitened data
    intsnr = get_snr(ts,T_obs,fs,psd.data.data,par.fmin)

    # normalise the waveform using either integrated or peak SNR
    intsnr = np.array(intsnr)
    scale = isnr/intsnr
    ts *= scale
    hp *= scale
    hc *= scale
    intsnr *= scale
    print '{}: computed the network SNR = {}'.format(time.asctime(),isnr)

    return ts, hp, hc

def gen_fs(fs,T_obs,par):
    """
    generates a BBH timedomain signal
    """
    N = T_obs * fs      # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    approximant = lalsimulation.IMRPhenomD
    f_high = fs/2.0
    df = 1.0/T_obs
    f_low = df*int(par.fmin/df)
    f_ref = f_low    
    dist = 1e6*lal.PC_SI  # put it as 1 MPc

    # make waveform
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
                    par.m1 * lal.MSUN_SI, par.m2 * lal.MSUN_SI,
                    0, 0, 0, 0, 0, 0,
                    dist,
                    par.iota, 
                    0, 0, 0, 0,
                    df,
                    f_low,f_high,
                    f_ref,
                    lal.CreateDict(),
                    approximant)

    return hp.data.data, hc.data.data

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
    ht_tck = interpolate.splrep(tvec, ht, s=0)
    hp_tck = interpolate.splrep(tvec, hp, s=0)
    hc_tck = interpolate.splrep(tvec, hc, s=0)
    tnew = tvec + tdelay
    new_ht = interpolate.splev(tnew, ht_tck, der=0,ext=1)
    new_hp = interpolate.splev(tnew, hp_tck, der=0,ext=1)
    new_hc = interpolate.splev(tnew, hc_tck, der=0,ext=1)

    return new_ht, new_hp, new_hc

def gen_ts(fs,T_obs,isnr=1.0,det='H1',par=None):
    """ 
    generates a randomly chosen timeseries from one of the available classes
    """
    N = T_obs * fs          # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)    
 
    # make psds and noise realisations   
    psd = gen_psd(fs,T_obs,op='AdvDesign',det=det)
    noise = gen_noise(fs,T_obs,psd.data.data)

    print '{}: making a BBH + noise instance'.format(time.asctime())
    sig, hp, hc = gen_bbh(fs,T_obs,psd,isnr,det,beta=[0.8,1.0],par=par)
    
    return sig, noise, hp, hc

def load_data(initial_dataset):
    # get core name of dataset
    print('Using data for: {0}'.format(initial_dataset))

    #load in dataset 0
    with open(initial_dataset, 'rb') as rfp:
        base_test_set = pickle.load(rfp)

    return base_test_set

# the main part of the code
def main():
    """
    The main code - reads in data and template bank and performs matched
    filtering analysis
    """

    # get the command line args
    args = parser()
    np.random.seed(args.seed)
    
    # redefine things for conciseness
    Tobs = safe*args.Tobs       # observation time
    fs = args.fsample           # sampling frequency
    isnr = args.isnr            # integrated SNR
    N = Tobs*fs                 # the total number of time samples
    n = N // 2 + 1              # the number of frequency bins
    beta = [0.75,0.95]          # the desired window for merger time in fractions of input Tobs
    tmp_bank = args.temp_bank

    # set path to file
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath(args.dataset, cur_path)    

    # make the psds
    psd = gen_psd(fs,Tobs,op='AdvDesign',det='H1').data.data
    wpsd = (2.0/fs)*np.ones(n)         # define effective PSD for whited data

    # compute indices defining 
    low_idx,high_idx = convert_beta(beta,fs,Tobs)

    # load template bank
    tmp_bank = np.array(EventTable.read(tmp_bank,
    format='ligolw.sngl_inspiral', columns=['mass1','mass2','eta','mchirp']))

    # load signal/noise dataset
    data = load_data(new_path)
    Nsig = data[0].shape[0]

    #chi_bool = True
    #chi_rho = []
    #if chi_bool == True:
    #    count = 0
    #    for idx in xrange(Nsig):
    #        par = gen_par(tmp_bank[0],fs,Tobs,beta=beta)
    #        if data[1][idx] == 0:
                # whitened first template
    #            if count == 0:
    #                temp_par = par
    #            else:
    #                temp_par = gen_par(tmp_bank[0],fs,Tobs,beta=beta)
    #            fhp, fhc = gen_fs(tmp_bank[0],Tobs,temp_par)

    #            fmin = get_fmin(par.M,par.eta,Tobs)

                # whiten frequency domain template
    #            wfhp = whiten_data(fhp,Tobs,fs,psd,flag='fd')
    #            wfhc = whiten_data(fhc,Tobs,fs,psd,flag='fd')

                # calculate chi distribution. For testing purposes only!
    #            chi_rho.append(snr_ts(data[0][idx][0],wfhp,wfhc,T_obs,fs,wpsd,fmin,flag='fd')[int(N/2)])
    #            count+=1
    #            print '{}: Chi Rho for signal {} = {}'.format(time.asctime(),idx,chi_rho[-1])

        # save list of chi rho for test purposes only
    #    pickle_out = open("%schirho_values.pickle" % basename, "wb")
    #    pickle.dump(chi_rho, pickle_out)
    #    pickle_out.close()

    # loop over signals
    maxSNRts = np.zeros(Nsig)     # store maximised (over time) measured signal SNR
    print '{}: starting to generate data'.format(time.asctime())
    for i in xrange(2500):
    
        ###-CHANGE-TO-READ-IN-TEST-DATA-#############################################################
        # read in whitened time domain data 
        # generate parameters and unwhitened timeseries
        par = gen_par(tmp_bank[0],fs,Tobs,beta=beta)
        #sig,noise,_,_ = gen_ts(fs,Tobs,isnr,'H1',par)
        #data = sig + noise
        #wdata = whiten_data(data,Tobs,fs,psd,flag='td')
        ##############################################################################################

        ##############################################################################################
        # loop over templates
        for k,_ in enumerate(tmp_bank):
        #for k in xrange(args.Ntemp):

            ###-CHANGE-TO-READ-IN-TEMPLATE-###########################################################
            # read in template bank mass parameters
            # fhp,fhc = ????
            # generate unwhitened frequency domain waveform
            if k==0:
                temp_par = par
            else:
                temp_par = gen_par(tmp_bank[k],fs,Tobs,beta=beta)
            fhp, fhc = gen_fs(fs,Tobs,temp_par)
            #print '{}: Generated random mass frequency domain template'.format(time.asctime())
            ##########################################################################################

            # compute lower cut-off freq for template based on chirp mass and Tobs
            fmin = get_fmin(par.M,par.eta,Tobs)
            #print '{}: Template fmin -> {}'.format(time.asctime(),fmin)

            # whiten frequency domain template
            wfhp = whiten_data(fhp,Tobs,fs,psd,flag='fd')
            wfhc = whiten_data(fhc,Tobs,fs,psd,flag='fd')
            #print '{}: Whitened frequcny domain h+(f) and hx(f)'.format(time.asctime())   

            ##########################################################################################
            # compute SNR timeseries using frequency domain template

            SNRts = snr_ts(data[0][i][0],wfhp,wfhc,Tobs,fs,wpsd,fmin,flag='fd')
            temp = np.max(SNRts[low_idx:high_idx])
            if temp>maxSNRts[i]: maxSNRts[i] = temp
        print '{}: maximised signal {} SNR (FD template) type {} = {}'.format(time.asctime(),i,data[1][i],maxSNRts[i])

    # seperate noise from signal
    noise = []
    signals = []
    for idx, i in enumerate(maxSNRts[:2500]):
        if data[1][idx] == 0:
            noise.append(i)
        if data[1][idx] == 1:
            signals.append(i)

    # make distribution plots
    nbins_sig = int(np.sqrt(len(signals)))
    nbins_noise = int(np.sqrt(len(noise)))
    temp = np.linspace(0,isnr+8,1000)
    plt.figure()
    plt.hist(signals,nbins_sig,normed=True,alpha=0.5,label='max-temp sig (FD)')
    plt.hist(noise,nbins_noise,normed=True,alpha=0.5,label='max-temp noise (FD)')
    plt.plot(temp,norm.pdf(temp,loc=isnr),'k',label='1-temp noise (expect)')
    plt.xlim([0,np.max(temp)]) 
    plt.legend(loc='upper right')
    plt.xlabel('measured SNR')
    plt.ylabel('p(SNR)')
    plt.savefig('%smf_template.png' % args.basename)

    plt.ylim(ymin=1e-3,ymax=10)
    plt.yscale('log', nonposy='clip')
    plt.savefig('%slog_mf_template.png' % args.basename)

    # save list of rho for test signals and test noise
    pickle_out = open("%srho_values.pickle" % args.basename, "wb")
    pickle.dump(maxSNRts, pickle_out)
    pickle_out.close() 

if __name__ == "__main__":
    main()
