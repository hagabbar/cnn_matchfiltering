#!usr/bin/env python

import numpy as np
import cPickle as pickle
import glob

nts = 2
dtype = 'astromass'
snrs = [1,2,3,4,5,6,7,8,9,10]
datapath ='../bbh_data'


for snr in snrs:
    flags = np.array([])
    test_datasets = sorted(glob.glob(
        '{0}/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_ts_*.sav'.format(datapath, snr, dtype)))
    for ds in test_datasets[:nts]:
        print(ds)
        with open(ds, 'rb') as rfp:
            f = pickle.load(rfp)[1]
        flags = np.concatenate((flags, f))
    print(flags.shape)
    np.save('./flags/flags_{0}_snr{1}.npy'.format(dtype, snr), flags)
