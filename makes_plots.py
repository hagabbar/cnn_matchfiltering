#!usr/bin/env python

import os
import glob
import re
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cPickle as pickle

from sklearn import metrics
from scipy.stats import chi2, ncx2
from scipy import interpolate, special


def sigma(n, tp):
    """

    :param tp:
    :param n:
    :return:
    """
    return np.sqrt(tp*(1.-tp)/n)


def CNN_ROC(CNN_path, snr, run):
    """
    calculate roc curve for cnn results
    """
    targets = np.load('{0}/SNR{1}/run{2}/targets.npy'.format(CNN_path,snr,run))[:,1]
    preds = np.load('{0}/SNR{1}/run{2}/preds.npy'.format(CNN_path,snr,run))[:,1]
    assert len(preds) == len(targets)
    FDP, TDP, _ = metrics.roc_curve(targets, preds)
    FDP_error = [sigma(len(targets)/2., t) for t in FDP]
    TDP_error = [sigma(len(targets)/2., t) for t in TDP]
    return FDP, TDP, FDP_error, TDP_error


def accuracy(FDP,TDP,TDP_error, threshold):
    """
    calculate accuracy(efficiency) given fdp and tdp
    """
    fx = interpolate.interp1d(FDP,TDP)
    fx_e_neg = interpolate.interp1d(FDP,TDP-TDP_error)
    fx_e_pos = interpolate.interp1d(FDP,TDP+TDP_error)
    acc = fx(threshold)
    acc_e_neg = fx_e_neg(threshold)
    acc_e_pos = fx_e_pos(threshold)
    return acc, acc_e_neg, acc_e_pos


def process_mf_results(mf_path, mf_seed, snr):
    """
    get snr values from mf analysis
    """
    dir = '{0}/results/full_bank_snr{1}/seed_{2}'.format(mf_path, snr, mf_seed)
    #print(dir)
    #
    rho_values = np.array([])

    for i in range(0,2):
        rho_files = sorted(glob.glob('{0}/ts_{1}/rho_values_*.pickle'.format(dir,i)))
        rho_files = sorted(rho_files, key=lambda a: int(re.split(r'[_-]+', a)[-2]))
        for rf in rho_files:
            with open(rf, 'rb') as rfp:
                rv = pickle.load(rfp)
            rho_values = np.concatenate((rho_values,rv))

    return rho_values


def load_flags(flags_path, snr, type):
    """
    load flags for dataset
    """
    return np.load('{0}/flags_{1}_snr{2}.npy'.format(flags_path, type, snr))


def mf_roc(cnn_path, snr, rho_values):

    flags = np.load('{0}/SNR{1}/run0/targets.npy'.format(cnn_path, snr))

    noise = rho_values[np.where(flags[:,0]==1)]
    signals = rho_values[np.where(flags[:,1]==1)]
    thresholds = np.linspace(0,12,1000)
    TP, _, _, FP = cal_roc(noise,signals, thresholds)
    FP_error = [sigma(len(signals), t) for t in FP]
    TP_error = [sigma(len(signals), t) for t in TP]

    return TP, FP, TP_error, FP_error

def cal_roc(noise, signals, thresholds):
    # lists to populate
    TP = []
    TN = []
    FP = []
    FN = []
    # loop over thresholds
    for t in thresholds:
        TP.append(len(signals[signals > t]))
        TN.append(len(noise[noise < t]))
        FN.append(len(signals[signals < t]))
        FP.append(len(noise[noise > t]))

    Nsig = float(len(signals))
    Nnoise = float(len(noise))

    TP = np.array(TP)/Nsig
    FN = np.array(FN)/Nsig
    TN = np.array(TN)/Nnoise
    FP = np.array(FP)/Nnoise

    return TP, FN, TN, FP



def interp_sig(snr, tdp, threshold, c):
    """
    use spline interpolation to fit a line
    uses sigmoid to map (0,1) to (-inf, inf) and vice-versa
    :param snr:
    :param rho:
    :param threshold:
    :return:
    """
    # assume start with NaN
    flag = True
    while flag:
        snr_new, tdp_new = interp_loop(snr, tdp, threshold, c)
        if not np.isnan(tdp_new).any():
            flag = False
        else:
            snr = snr[1:]
            tdp = tdp[1:]

    if np.isnan(tdp_new).any():
        print('Failed to avoid NaNs')

    return snr_new, tdp_new

def interp_loop(snr, tdp, threshold, c):
    """
    function to loop over to peform spline interpolation
    :param snr:
    :param rho:
    :param threshold:
    :param c: fuzzfactor
    """
    # find min value
    min_tdp = np.min(tdp)
    # take log of data to avoid negative values
    tmp_tdp = special.logit(tdp - min_tdp + c)
    # interpolate with spline interpolation
    tck = interpolate.splrep(snr, tmp_tdp)
    # new x and y values
    snr_new = np.linspace(1, 10, 1e3)
    tmp_tdp_new = interpolate.splev(snr_new, tck, der=0)
    # return to linear space
    tdp_new = special.expit(tmp_tdp_new) + min_tdp - c

    return snr_new, tdp_new

# path to MF results
mf_path = './final_mf_results'
# CNN path
cnn_path = './final_runs'

# path to flags (previously generated files)
flags_path = './flags'
type = 'astromass'

# fuzz factors for each thresholds
ffs = [0.5e-3, 0.5e-3, 0.5e-3]

snrs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
runs = [1,1,1,1,1,1,1,3,1,1]
mf_snrs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mf_seeds = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
thresholds = [0.1, 0.01, 0.001]

# empty lists to populate
MF_ROCs = []
CNN_ROCs = []
MF_acc = []
CNN_acc = []

# calcualte ROC curves and accuracy
for snr, run in zip(snrs, runs):
    FDP, TDP, FDP_error, TDP_error = CNN_ROC(cnn_path, snr, run)
    CNN_ROCs.append([FDP, TDP, FDP_error, TDP_error])
    CNN_acc.append([accuracy(FDP, TDP, TDP_error, t) for t in thresholds])

for snr, mf_seed in zip(mf_snrs ,mf_seeds):
    rho_values = process_mf_results(mf_path, mf_seed, snr)
    flags = load_flags(flags_path, snr, type)
    MF_TDP, MF_FDP, MF_TDP_error, MF_FDP_error = mf_roc(cnn_path, snr, rho_values)

    MF_ROCs.append([MF_FDP, MF_TDP, MF_FDP_error, MF_TDP_error])
    MF_acc.append([accuracy(MF_FDP, MF_TDP, MF_TDP_error, t) for t in thresholds])

# colours and line styles for plots
colours = ['indigo', 'c', 'darkorange', 'indigo', 'c', 'darkorange', 'indigo', 'c', 'darkorange']
linestyles = ['-', '--', '-.', ':']

lineArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-')
dashArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')
dashdotArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-.')
dotArtist = plt.Line2D((0,1), (0, 0), color = 'k', linestyle = ':')
c1Artist = plt.Line2D((0, 1), (0, 0), color=colours[0], linestyle='', marker='o')
c2Artist = plt.Line2D((0, 1), (0, 0), color=colours[1], linestyle='', marker='o')
c3Artist = plt.Line2D((0, 1), (0, 0), color=colours[2], linestyle='', marker='o')

# handles for different plots
handles = [c1Artist, c2Artist, lineArtist, dashArtist, dashdotArtist]
acc_handles = [c1Artist, c2Artist, lineArtist, dashArtist, dashdotArtist]
ROC_labels = ['CNN', 'Matched filtering', 'SNR2', 'SNR4', 'SNR6']
acc_labels = ['CNN', 'Matched filtering', 'FAP = 0.1', 'FAP = 0.01', 'FAP = 0.001']

# plot roc curves
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_rasterization_zorder(1)

cmap_colours = plt.cm.jet_r(np.linspace(0, 1, 10))

# choose snrs to plot
to_plot = [1,3,5]

for i, d in enumerate([CNN_ROCs[n] for n in to_plot]):
    ax1.plot(d[0], d[1], c=colours[0], linestyle=linestyles[i], label='CNN snr {0}'.format(snrs[i]))
    ax1.fill_between(d[0], d[1]+d[3], d[1]-d[3], alpha=0.2, facecolor=colours[0], zorder=0)

for i, d in enumerate([MF_ROCs[n] for n in to_plot]):
    ax1.plot(d[0], d[1], c=colours[1], linestyle=linestyles[i], label='mf snr {0}'.format(mf_snrs[i]))
    ax1.fill_between(d[0], d[1]+d[3], d[1]-d[3], alpha=0.2, facecolor=colours[1], zorder=0)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1e-4, 1)
ax1.set_ylim(1e-4, 1)
ax1.legend(handles, ROC_labels)
ax1.set_xlabel('False alarm probability')
ax1.set_ylabel('True alarm probability')

# save all possible versions
fig1.savefig('ROC_curves.png', dpi=1200)
fig1.savefig('ROC_curves.pdf', dpi=1200)
fig1.savefig('ROC_curves.eps', rasterized=True, dpi=1200)
fig1.savefig('ROC_curves400.eps', rasterized=True, dpi=400)

# plot efficiency curve
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1, )
ax2.set_rasterization_zorder(1)

for i in range(len(thresholds)):
    # points to plot
    cnn = np.asarray([a[i][0] for a in CNN_acc])
    # line fit
    snr_cnn, eff_cnn = interp_sig(snrs, cnn, thresholds[i], ffs[i])
    # points to plot
    mf = np.asarray([a[i][0] for a in MF_acc])
    # line fit
    snr_mf, eff_mf = interp_sig(snrs, mf, thresholds[i], ffs[1])

    # errors
    d_eff_cnn = [sigma(1e4, e) for e in eff_cnn]
    d_eff_mf = [sigma(1e4, e) for e in eff_mf]

    # plot points for cnn
    ax2.plot(snrs, cnn, c=colours[0], marker = 'o', linestyle='')
    # plot line fit for cnn
    ax2.plot(snr_cnn, eff_cnn, c=colours[0], linestyle=linestyles[i])
    # plot error region for cnn
    ax2.fill_between(snr_cnn, eff_cnn - d_eff_cnn , eff_cnn + d_eff_cnn, facecolor=colours[0], alpha=0.2, zorder=0)

    # plot points for mf
    ax2.plot(mf_snrs, mf, c=colours[1], marker = 'o', linestyle='')
    # plot line fit for mf
    ax2.plot(snr_mf, eff_mf, c=colours[1], linestyle=linestyles[i])
    # plot error region for mf
    ax2.fill_between(snr_mf, eff_mf - d_eff_mf, eff_mf + d_eff_mf, facecolor=colours[1], alpha=0.2, zorder=0)

ax2.legend(acc_handles, acc_labels)
ax2.set_xticks(snrs)
ax2.grid()
ax2.set_xlabel(r'$\rho_{\mathrm{opt}}$')
ax2.set_ylabel('True alarm probability')
ax2.set_xlim(snrs[0], snrs[-1])
ax2.set_ylim(0, 1)

# save all possible versions
fig2.savefig('efficiency.png',  dpi=1200)
fig2.savefig('efficiency.pdf',  dpi=1200)
fig2.savefig('efficiency.eps', rasterized=True, dpi=1200)
fig2.savefig('efficiency400.eps', rasterized=True, dpi=400)
