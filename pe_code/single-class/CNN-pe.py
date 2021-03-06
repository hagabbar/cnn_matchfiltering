#!/usr/bin/env python

from __future__ import print_function, division
import argparse
import shutil
import keras
from keras.models import Sequential
import numpy as np
np.random.seed(1234)
import cPickle as pickle
import h5py
from keras.layers import Conv2D, MaxPool2D,Dense, Activation, Dropout, GaussianDropout, ActivityRegularization, Flatten
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.activations import softmax, relu, elu
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import to_categorical
import os, sys, shutil
import glob
from math import exp, log
#import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model

from clr_callback import *
from sklearn import preprocessing

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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
    """
    Parses command line arguments
    :return: arguments
    """

    #TODO: complete help sections

    parser = argparse.ArgumentParser(prog='CNN-keras.py', description='Convolutional Neural Network in keras with tensorflow')

    # arguments for data
    parser.add_argument('-SNR', '--SNR', type=int,
                        help='')
    parser.add_argument('-trdt', '--training_dtype', type=str,
                        help='')
    parser.add_argument('-tsdt', '--testing_dtype', type=str,
                        help='')
    parser.add_argument('--datapath', type=str,
                        help='')
    parser.add_argument('-Nts', '--Ntimeseries', type=int, default=10000,
                        help='number of time series for training')
    #parser.add_argument('-ds', '--set-seed', type=str,
    #                    help='seed number for each training/validaiton/testing set')
    parser.add_argument('-Ntot', '--Ntotal', type=int, default=10,
                        help='number of available datasets with the same name as specified dataset')
    parser.add_argument('-Nval', '--Nvalidation', type=int, default=10000,
                        help='')

    # arguments for input data to network (e.g training/testing/validation data/params)
    parser.add_argument('-Trd', '--training_dataset', type=str,
                       default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_ts.sav', 
                       help='path to the data')
    parser.add_argument('-Trp', '--training_params', type=str, #nargs='+',
                       default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_params.sav',
                       help='path to the training params')
    parser.add_argument('-Vald', '--validation_dataset', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_ts.sav', 
                        help='path to the data')
    parser.add_argument('-Valp', '--validation_params', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_params.sav',
                        help='path to the validation params')
    parser.add_argument('-Tsd', '--test_dataset', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_ts.sav', 
                        help='path to the data')
    parser.add_argument('-Tsp', '--test_params', type=str,
                        default='./deepdata_bbh/BBH_1s_8192Hz_3K_iSNR10_z1_params.sav',
                        help='path to the testing params')


    parser.add_argument('-bs', '--batch_size', type=int, default=20,
                        help='size of batches used for training/validation')
    parser.add_argument('-nw', '--noise_weight', type=float, default=1.0,
                        help='')
    parser.add_argument('-sw', '--sig_weight', type=float, default=1.0,
                        help='')
    # arguments for optimizer
    parser.add_argument('-opt', '--optimizer', type=str, default='SGD',
                        help='')
    parser.add_argument('-lr', '--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('-mlr', '--max_learning_rate', type=float, default=0.01,
                        help='max learning rate for cyclical learning rates')
    parser.add_argument('-NE', '--n_epochs', type=int, default=20,
                        help='number of epochs to train for')
    parser.add_argument('-dy', '--decay', type=float ,default=0.0,
                        help='help')
    parser.add_argument('-ss', '--stepsize', type=float, default=500,
                        help='help')
    parser.add_argument('-mn', '--momentum', type=float, default=0.9,
                        help='momentum for updates where applicable')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='')
    parser.add_argument('--epsilon', type=float, default=1e-08,
                        help='')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='')
    parser.add_argument('-pt', '--patience', type=int, default=10,
                        help='')
    parser.add_argument('-lpt', '--LRpatience', type=int, default=5,
                        help='')

    #arguments for network
    parser.add_argument('-f', '--features', type=str, default="1,1,1,1,0,4" ,
                        help='order and types of layers to use, see RunCNN_bbh.sh for types')
    parser.add_argument('-nf', '--nfilters', type=str, default="16,32,64,128,32,2",
                        help='number of kernels/neurons per layer')
    parser.add_argument('-fs', '--filter_size', type=str, default="1-1-32,1-1-16,1-1-8,1-1-4,0-0-0,0-0-0" ,
                        help='size of convolutional layers')
    parser.add_argument('-fst', '--filter_stride', type=str, default="1-1-1,1-1-1,1-1-1,1-1-1",
                        help='stride for max-pooling layers')
    parser.add_argument('-fpd', '--filter_pad', type=str, default="0-0-0,0-0-0,0-0-0,0-0-0",
                        help='padding for convolutional layers')
    parser.add_argument('-dl', '--dilation', type=str, default="1-1-1,1-1-1,1-1-4,1-1-4,1-1-1",
                        help='dilation for convolutional layers, set to 1 for normal convolution')
    parser.add_argument('-p', '--pooling', type=str, default="1,1,1,1",
                        help='')
    parser.add_argument('-ps', '--pool_size', type=str, default="1-1-8,1-1-6,1-1-4,1-1-2",
                        help='size of max-pooling layers after convolutional layers')
    parser.add_argument('-pst', '--pool_stride', type=str, default="1-1-4,1-1-4,1-1-4,0-0-0,0-0-0",
                        help='stride for max-pooling layers')
    parser.add_argument('-ppd', '--pool_pad', type=str, default="0-0-0,0-0-0,0-0-0",
                        help='')
    parser.add_argument('-dp', '--dropout', type=str, default="0.0,0.0,0.0,0.0,0.1,0.0",
                        help='dropout for the fully connected layers')
    parser.add_argument('-fn', '--activation_functions', type=str, default='elu,elu,elu,elu,elu,softmax',
                        help='activation functions for layers')

    # general arguments
    parser.add_argument('-od', '--outdir', type=str, default='./history',
                        help='')
    parser.add_argument('--notes', type=str,
                        help='')

    return parser.parse_args()


class network_args:
    def __init__(self, args):
        self.features = np.array(args.features.split(','))
        self.num_classes = 1 
        self.class_weight = {0:args.noise_weight, 1:args.sig_weight}
        self.Nfilters = np.array(args.nfilters.split(",")).astype('int')
        self.kernel_size = np.array([i.split("-") for i in np.array(args.filter_size.split(","))]).astype('int')
        self.stride = np.array([i.split("-") for i in np.array(args.filter_stride.split(","))]).astype('int')
        self.dilation = np.array([i.split("-") for i in np.array(args.dilation.split(","))]).astype('int')
        self.activation = np.array(args.activation_functions.split(','))
        self.dropout = np.array(args.dropout.split(",")).astype('float')
        self.pooling = np.array(args.pooling.split(',')).astype('bool')
        self.pool_size = np.array([i.split("-") for i in np.array(args.pool_size.split(","))]).astype('int')
        self.pool_stride = np.array([i.split("-") for i in np.array(args.pool_stride.split(","))]).astype('int')



def choose_optimizer(args):

    lr = args.lr

    if args.optimizer == 'SGD':
        return SGD(lr=lr, momentum=args.momentum, decay=args.decay, nesterov=args.nesterov)
    if args.optimizer == 'RMSprop':
        return RMSprop(lr=lr, rho=args.rho, epsilon=args.epsilon, decay=args.decay)
    if args.optimizer == 'Adagrad':
        return Adagrad(lr=lr, epsilon=args.epsilon, decay=args.decay)
    if args.optimizer == 'Adadelta':
        return Adadelta(lr=lr, rho=args.rho, epsilon=args.epsilon, decay=args.decay)
    if args.optimizer =='Adam':
        return Adamax(lr=lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon, decay=args.decay)
    if args.optimizer == 'Adamax':
        return Adam(lr=lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon, decay=args.decay)
    if args.optimizer =='Nadam':
        return Nadam(lr=lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon, schedule_decay=args.decay)


def network(args, netargs, shape, outdir, x_train, y_train, x_val, y_val, x_test, y_test, samp_weights=None):

    model = Sequential()

    optimizer = choose_optimizer(args)

    #TODO: add support for advanced activation functions

    for i, op in enumerate(netargs.features):

        if int(op) == 1:
            # standard convolutional layer with max pooling
            model.add(Conv2D(
                netargs.Nfilters[i],
                input_shape=shape,
                kernel_size=netargs.kernel_size[i],
                strides= netargs.stride[i],
                padding= 'valid',
                data_format='channels_first',
                dilation_rate=netargs.dilation[i],
                use_bias=True,
                kernel_initializer=initializers.glorot_normal(),
                bias_initializer=initializers.glorot_normal(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            ))

            if netargs.activation[i] == 'leakyrelu':
                model.add(LeakyReLU(alpha=0.01))#netargs.activation[i]))
            elif netargs.activation[i] == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(netargs.activation[i]))

            model.add(BatchNormalization(
                axis=1
            ))

            model.add(GaussianDropout(netargs.dropout[i]))

            if netargs.pooling[i]:
                model.add(MaxPool2D(
                    pool_size=netargs.pool_size[i],
                    strides=netargs.pool_stride[i],
                    padding='valid',
                    data_format='channels_first'
                ))

        elif int(op) == 0:
            # standard fully conected layer
            model.add(Flatten())
            model.add(Dense(
                netargs.Nfilters[i]
                #kernel_regularizer=regularizers.l1(0.01)
            ))

            if netargs.activation[i] == 'leakyrelu':
                model.add(LeakyReLU(alpha=0.01))#netargs.activation[i]))
            elif netargs.activation[i] == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(netargs.activation[i]))

            model.add(GaussianDropout(netargs.dropout[i]))

        elif int(op) == 2:
            # standard fully conected layer
            #model.add(Flatten())
            model.add(Dense(
                netargs.Nfilters[i]
                #kernel_regularizer=regularizers.l1(0.01)
            ))

            if netargs.activation[i] == 'leakyrelu':
                model.add(LeakyReLU(alpha=0.01))#netargs.activation[i]))
            elif netargs.activation[i] == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(netargs.activation[i]))

            model.add(GaussianDropout(netargs.dropout[i]))

        elif int(op) == 4:
            # softmax output layer
            model.add(Dense(
                netargs.num_classes
            ))
            if netargs.activation[i] == 'leakyrelu':
                model.add(LeakyReLU(alpha=0.01))#netargs.activation[i]))
            elif netargs.activation[i] == 'prelu':
                model.add(PReLU())
            else:
                model.add(Activation(netargs.activation[i]))

    print('Compiling model...')

    model.compile(
        loss="logcosh",
        optimizer=optimizer,
        metrics=["accuracy", "categorical_crossentropy"]
    )

    model.summary()

    #TODO: add options to enable/disable certain callbacks


    clr = CyclicLR(base_lr=args.lr, max_lr=args.max_learning_rate, step_size=args.stepsize)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=0, mode='auto')

    redLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=args.LRpatience, verbose=0, mode='auto',
                              epsilon=0.0001, cooldown=0, min_lr=0)

    modelCheck = ModelCheckpoint('{0}/best_weights.hdf5'.format(outdir), monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=True, mode='auto', period=0)

    print('Fitting model...')
    if args.lr != args.max_learning_rate:
        hist = model.fit(x_train, y_train,
                         epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         #class_weight=netargs.class_weight,
                         #sample_weight=samp_weights,
                         validation_data=(x_val, y_val),
                         shuffle=True,
                         verbose=1,
                         callbacks=[clr, earlyStopping, redLR, modelCheck])
    else:
        hist = model.fit(x_train, y_train,
                         epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         #class_weight=netargs.class_weight,
                         #sample_weight=samp_weights,
                         validation_data=(x_val, y_val),
                         shuffle=True,
                         verbose=1,
                         callbacks=[earlyStopping, modelCheck])

    print('Evaluating model...')

    model.load_weights('{0}/best_weights.hdf5'.format(outdir))

    eval_results = model.evaluate(x_test, y_test,
                                  sample_weight=None,
                                  batch_size=args.batch_size, verbose=1)

    preds = model.predict(x_test)

    return model, hist, eval_results, preds


def concatenate_datasets(datapath, snr, training_dtype, testing_dtype, Nts, Nval = 10000, Ntot = 30):
    """
    shorten and concatenate data
    :param initial_dataset: first dataset in the set
    :param Nts: total number of images/time series
    :param Ntot: total number of available datasets
    :return:
    """


    print('Using data located in: {0}'.format(datapath))
    training_datasets = sorted(glob.glob('{0}/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR{1}_Hdet_{2}_*seed_ts_*.sav'.format(datapath, snr, training_dtype)))
    validation_datasets = sorted(glob.glob('{0}/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_ts_*.sav'.format(datapath, snr, testing_dtype)))
    test_datasets = sorted(glob.glob('{0}/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_ts_*.sav'.format(datapath, snr, testing_dtype)))
    #print(training_datasets, validation_datasets, test_datasets)


    print('Using data located in: {0}'.format(datapath))
    training_paramsets = sorted(glob.glob('{0}/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR{1}_Hdet_{2}_*seed_params_*.sav'.format(datapath, snr, training_dtype)))
    validation_paramsets = sorted(glob.glob('{0}/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_params_*.sav'.format(datapath, snr, testing_dtype)))
    test_paramsets = sorted(glob.glob('{0}/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR{1}_Hdet_{2}_*seed_params_*.sav'.format(datapath, snr, testing_dtype)))
    #print(training_paramsets, validation_paramsets, test_paramsets)

    # load in dataset 0 params and labels
    with open(training_datasets[0], 'rb') as rfp, open(training_paramsets[0], 'rb') as p:
        base_train_set = pickle.load(rfp)[0]
        base_train_par = np.array(pickle.load(p))
        base_train_set = [base_train_set, base_train_par]

        # set noise sample param values to zero
        # get desired parameter
        base_train_new = False
        for idx,i in enumerate(base_train_set[1]):
            if i != None and not base_train_new:
                base_train_new = [[base_train_set[0][idx]],[np.array([i.mc])]]
            elif i != None and base_train_new:
                base_train_new[1].append(np.array([i.mc]))
                base_train_new[0].append(base_train_set[0][idx])
        base_train_set = [np.array(base_train_new[0]),np.array(base_train_new[1])]

    with open(validation_datasets[0], 'rb') as rfp, open(validation_paramsets[0], 'rb') as p:
        base_valid_set = pickle.load(rfp)[0]
        base_valid_par = np.array(pickle.load(p))
        base_valid_set = [base_valid_set, base_valid_par]

        # set noise sample param values to zero
        # get desired parameter
        base_valid_new = False
        for idx,i in enumerate(base_valid_set[1]):
            if i != None and not base_valid_new:
                base_valid_new = [[base_valid_set[0][idx]],[np.array([i.mc])]]
            elif i != None and base_valid_new:
                base_valid_new[1].append([np.array([i.mc])])
                base_valid_new[0].append(base_valid_set[0][idx])
        base_valid_set = [np.array(base_valid_new[0]),np.array(base_valid_new[1])]

    with open(test_datasets[0], 'rb') as rfp, open(test_paramsets[0], 'rb') as p:
        base_test_set = pickle.load(rfp)[0]
        base_test_par = np.array(pickle.load(p))
        base_test_set = [base_test_set, base_test_par]

        # set noise sample param values to zero
        # get desired parameter
        base_test_new = False
        for idx,i in enumerate(base_test_set[1]):
            if i != None and not base_test_new:
                base_test_new = [[base_test_set[0][idx]],[np.array([i.mc])]]
            elif i != None and base_test_new:
                base_test_new[1].append([np.array([i.mc])])
                base_test_new[0].append(base_test_set[0][idx])
        base_test_set = [np.array(base_test_new[0]),np.array(base_test_new[1])]


    # size of data sets
    size = int(1e4)
    val_size = int(1e4)
    # number of datasets -  depends on Nts
    Nds = np.floor(Nts / float(size))
    print(Nds)
    # check there are sufficient datasets
    if not Nds <= Ntot:
        print('Error: Insufficient datasets for number of time series')
        exit(0)

    # start with training set
    # if more than the initial data set is needed
    if Nds > 1:
        # how many images/time series needed
        need = Nts - size

        # loop over enough files to reach total number of time series
        for ps_idx,ds in enumerate(training_datasets[1:int(Nds)]):
            with open(ds, 'rb') as rfp, open(training_paramsets[ps_idx+1], 'rb') as p:
                train_set = pickle.load(rfp)[0]
                train_par = np.array(pickle.load(p))
                train_set = [train_set, train_par]


                # set noise sample param values to zero
                # get desired parameter
                train_new = False
                for idx,i in enumerate(train_set[1]):
                    if i != None and not train_new:
                        train_new = [[train_set[0][idx]],[np.array([i.mc])]]
                    elif i != None and train_new:
                        train_new[1].append([np.array([i.mc])])
                        train_new[0].append(train_set[0][idx])
                train_set = [np.array(train_new[0]),np.array(train_new[1])]

            # check if this set needs truncating
            if need > size:
                cut = size
            else:
                cut = need

            # empty arrays to populate
            aug_train_set = np.zeros(2, dtype = np.ndarray) # change to two if wanting labels as well
            # concatenate the arrays
            for i in range(2): # change to 2 if also wanting labels
                aug_train_set[i] = np.concatenate((base_train_set[i], train_set[i][:cut]), axis=0)
            # copy as base set for next loop
            base_train_set = aug_train_set


            need -= size


    else:
        # return truncated version of the initial data set
        # change both 1 numbers back to 2 if wanting labels as well
        aug_train_set = np.zeros(2, dtype=np.ndarray)

        for i in range(2):
            aug_train_set[i] = base_train_set[i][:Nts]

        base_train_set = aug_train_set

    # validation/testing fixed at 10K
    Nds_val = np.floor(Nval / float(val_size))
    # check there are sufficient datasets
    if not Nds_val <= Ntot:
        print('Error: Insufficient datasets for number of time series')
        exit(0)

    if Nds_val > 1:
        # how many images/time series needed
        need = Nval - val_size


        # loop over enough files to reach total number of time series
        for Vds, Vps, Tds, Tps in zip(validation_datasets[1:int(Nds_val)], validation_paramsets[1:int(Nds_val)], test_datasets[1:int(Nds_val)], test_paramsets[1:int(Nds_val)]):
            with open(Vds, 'rb') as rfp, open(Vps, 'rb') as p:
                valid_set = pickle.load(rfp)[0]
                valid_params = np.array(pickle.load(p))
                valid_set = [valid_set,valid_params]

                # set noise sample param values to zero
                # get desired parameter
                valid_new = False
                for idx,i in enumerate(valid_set[1]):
                    if i != None and not valid_new:
                        valid_new = [[valid_set[0][idx]],[np.array([i.mc])]]
                    elif i != None and valid_new:
                        valid_new[1].append([np.array([i.mc])])
                        valid_new[0].append(valid_set[0][idx])
                valid_set = [np.array(valid_new[0]),np.array(valid_new[1])]

            with open(Tds, 'rb') as rfp, open(Tps, 'rb') as p:
                test_set = pickle.load(rfp)[0]
                test_params = np.array(pickle.load(p))
                test_set = [test_set,test_params]

                # set noise sample param values to zero
                # get desired parameter
                test_new = False
                for idx,i in enumerate(test_set[1]):
                    if i != None and not test_new:
                        test_new = [[test_set[0][idx]],[np.array([i.mc])]]
                    elif i != None and test_new:
                        test_new[1].append([np.array([i.mc])])
                        test_new[0].append(test_set[0][idx])
                test_set = [np.array(test_new[0]),np.array(test_new[1])]

            # check if this set needs truncating
            if need > val_size:
                cut = val_size
            else:
                cut = need

            # empty arrays to populate
            aug_valid_set = np.zeros(2, dtype = np.ndarray) # change back to 2 for labels
            aug_test_set = np.zeros(2, dtype=np.ndarray) # change back to 2 for labels
            # concatenate the arrays
            for i in range(2): # change back to 2 for labels
                aug_valid_set[i] = np.concatenate((base_valid_set[i], valid_set[i][:cut]), axis=0)
                aug_test_set[i] = np.concatenate((base_test_set[i], test_set[i][:cut]), axis=0)

            # copy as base set for next loop
            base_valid_set = aug_valid_set
            base_test_set = aug_test_set

            need -= val_size


    else:
        # return truncated version of the initial data set
        aug_valid_set = np.zeros(2, dtype=np.ndarray) # change back to 2 for labels
        aug_test_set = np.zeros(2, dtype=np.ndarray) # change back to 2 for labels

        for i in range(2): # change back to 2 for labels
            aug_valid_set[i] = base_valid_set[i][:Nval]
            aug_test_set[i] = base_test_set[i][:Nval]

        base_valid_set = aug_valid_set
        base_test_set = aug_test_set


    return base_train_set, base_valid_set, base_test_set


def truncate_dataset(dataset, start, length):
    """

    :param dataset:
    :param start:
    :param end:
    :return:
    """
    print('    length of data prior to truncating: {0}'.format(dataset[0].shape))
    print('    truncating data between {0} and {1}'.format(start, start+length))
    # shape of truncated dataset
    new_shape = (dataset[0].shape[0],1,length)
    # array to populate
    #truncated_data = np.empty(new_shape, dtype=np.ndarray)
    # loop over data and truncate
    #for i,ts in enumerate(dataset[0]):
    #    truncated_data[i] = ts[0,start:(start+length)].reshape(1,length)

    dataset[0] = dataset[0][:,:,start:(start+length)]
    print('    length of truncated data: {}'.format(dataset[0].shape))
    return dataset



def load_data(args, netargs):
    """
    Load the data set
    :param dataset: the path to the data set (string)
    :param Nts: total number of time series for training
    :return: tuple of theano data set
    """

    train_set, valid_set, test_set = concatenate_datasets(args.datapath, args.SNR, args.training_dtype,
                                                          args.testing_dtype, args.Ntimeseries, args.Nvalidation, args.Ntotal)


    start = 4096
    length = 8192
    print('Truncating training set')
    train_set = truncate_dataset(train_set,start, length)
    print('Truncating validation set')
    valid_set = truncate_dataset(valid_set,start, length)
    print('Truncating test set')
    test_set = truncate_dataset(test_set, start, length)

    Ntrain = train_set[0].shape[0]
    xshape = train_set[0].shape[1]
    yshape = train_set[0].shape[2]
    channels = 1

    rescale = False

    if rescale:
        print('Rescaling data')
        for i in range(Ntrain):
            train_set[0][i] = preprocessing.normalize(train_set[0][i])

        for i in range(args.Nvalidation):
            valid_set[0][i] = preprocessing.normalize(valid_set[0][i])
            test_set[0][i] = preprocessing.normalize(test_set[0][i])

    x_train = (train_set[0].reshape(Ntrain, channels, xshape, yshape))
    #y_train = to_categorical(train_set[1], num_classes=netargs.num_classes)
    y_train = train_set[1]
    x_val = (valid_set[0].reshape(valid_set[0].shape[0], channels, xshape, yshape))
    #y_val = to_categorical(valid_set[1], num_classes=netargs.num_classes)
    y_val = valid_set[1]
    x_test = (test_set[0].reshape(test_set[0].shape[0], channels, xshape, yshape))
    #y_test = to_categorical(test_set[1], num_classes=netargs.num_classes)
    y_test = test_set[1]

    print('Traning set dimensions: {0}'.format(x_train.shape))
    print('Validation set dimensions: {0}'.format(x_val.shape))
    print('Test set dimensions: {0}'.format(x_test.shape))

    #return x_train, np.log(y_train), x_val, np.log(y_val), x_test, np.log(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def main(args):
    # get arguments
    # convert args to correct format for network
    netargs = network_args(args)

    # load in training set weighting parameters
    #for idx,file in enumerate(glob.glob('%s*' % args.training_params)):
    #    if idx == 0:
    #        with open(file, 'rb') as rfp:
    #            tr_params = np.array(pickle.load(rfp))
    #    else:
    #        with open(file, 'rb') as rfp:
    #            tr_params = np.append(tr_params,np.array(pickle.load(rfp)))

    # calculate unormalized weighting vector
    #final_tr_params = []
    #sig_params = []
    #for samp in tr_params:
    #    if samp == None:
    #        final_tr_params.append(1)
    #    elif samp != None:
    #        final_tr_params.append(samp.mc**(-5.0/3.0))
    #        sig_params.append(samp.mc**(-5.0/3.0))


    #if samp != None:
        # normalize weighting vector
   #     sig_params /= np.max(np.abs(np.array(sig_params)),axis=0)
   #     sig_params *= 1e0
        #sig_params = np.array(sig_params)
   #     count = 0
   #     final_tr_params = []
   #     for samp in tr_params:
   #         if samp == None:
   #             final_tr_params.append(1/np.max(np.abs(np.array(sig_params)),axis=0))
   #         if samp != None:
   #             final_tr_params.append(sig_params[count]) 
   #             count += 1

   # final_tr_params = np.array(final_tr_params)

    # load in time series info
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args, netargs)

    if not os.path.exists('{0}/SNR{1}'.format(args.outdir,args.SNR)):
        os.makedirs('{0}/SNR{1}'.format(args.outdir,args.SNR))

    Nrun = 0
    while os.path.exists('{0}/SNR{1}/run{2}'.format(args.outdir,args.SNR,Nrun)):
        Nrun += 1
    os.makedirs('{0}/SNR{1}/run{2}'.format(args.outdir, args.SNR, Nrun))

    shape = x_train.shape[1:]
    out = '{0}/SNR{1}/run{2}'.format(args.outdir, args.SNR,Nrun)

    # train and test network
    model, hist, eval_results, preds = network(args, netargs, shape, out,
                                               x_train, y_train, x_val, y_val, x_test, y_test) # add final_tr_params at end if weighitng on params


    with open('{0}/SNR{1}/run{2}/args.pkl'.format(args.outdir, args.SNR, Nrun), "wb") as wfp:
        pickle.dump(args, wfp)

    for m,r in zip(model.metrics_names, eval_results):
        print('Test {0}: {1}'.format(m, r))

    #shutil.copy('./runCNN.sh', '{0}/SNR{1}/run{2}'.format(args.outdir, args.SNR,Nrun))

    model.save('{0}/SNR{1}/run{2}/nn_model.hdf5'.format(args.outdir,args.SNR,Nrun))

    with open('{0}/SNR{1}/run{2}/targets.pkl'.format(args.outdir,args.SNR,Nrun), 'wb') as output_file:
        pickle.dump(y_test, output_file)
    with open('{0}/SNR{1}/run{2}/preds.pkl'.format(args.outdir,args.SNR,Nrun), 'wb') as output_file:
        pickle.dump(preds, output_file)
    with open('{0}/SNR{1}/run{2}/history.pkl'.format(args.outdir,args.SNR,Nrun), 'wb') as output_file:
        pickle.dump(hist.history, output_file)
    with open('{0}/SNR{1}/run{2}/test_results.pkl'.format(args.outdir,args.SNR,Nrun), 'wb') as output_file:
        pickle.dump(eval_results, output_file)

    # np.save does not work with python 2.7.5
    #np.save('{0}/SNR{1}/run{2}/targets.npy'.format(args.outdir,args.SNR,Nrun),np.array(y_test))
    #np.save('{0}/SNR{1}/run{2}/preds.npy'.format(args.outdir,args.SNR,Nrun), preds)
    #np.save('{0}/SNR{1}/run{2}/history.npy'.format(args.outdir,args.SNR,Nrun), hist.history)
    #np.save('{0}/SNR{1}/run{2}/test_results.npy'.format(args.outdir,args.SNR,Nrun),eval_results)

    print('Results saved at: {0}/SNR{1}/run{2}'.format(args.outdir,args.SNR,Nrun))

if __name__ == '__main__':
    args = parser()
    main(args)
