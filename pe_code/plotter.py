import os,sys
import glob
import re
import numpy as np
import argparse
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

import cPickle as pickle

from sklearn import metrics
from scipy.stats import chi2, ncx2
from scipy.interpolate import interp1d

class PEresults:
    def __init__(self,Nsignal,Nnoise,SNR):
        self.Nsignal = Nsignal
        self.Nnoise = Nnoise
        self.SNR = SNR

# directory to use
dir = './history'
dir2 = './dgx_history'
snr = 8
runs = [10]
runs2 = [3,4]

for i,r in enumerate(runs):
    with open('{0}/SNR{1}/run{2}/targets.pkl'.format(dir,snr,r), 'rb') as rft, open('{0}/SNR{1}/run{2}/preds.pkl'.format(dir,snr,r)) as rfp:
        targets = pickle.load(rft)
        preds = pickle.load(rfp) 
        plt.scatter(targets,preds,s=2)
        plt.savefig('pe_scatter.png', dpi=1200, bbox_inches='tight')
        plt.close()
        sys.exit()
    targets = np.load('{0}/SNR{1}/run{2}/targets.npy'.format(dir,snr,r))[:,1]
    preds = np.load('{0}/SNR{1}/run{2}/preds.npy'.format(dir,snr,r))[:,1]
    print(targets)
    sys.exit()
