#!/usr/bin/env python

import argparse
import shutil
import keras
from keras.models import Sequential
import numpy as np
np.random.seed(1234)
import cPickle as pickle
import h5py
from keras.layers import Conv1D, MaxPool1D,Dense, Activation, Dropout, GaussianDropout, ActivityRegularization, Flatten
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
    parser.add_argument('-Nts', '--Ntimeseries', type=int, default=10000,
                        help='number of time series for training')
    #parser.add_argument('-ds', '--set-seed', type=str,
    #                    help='seed number for each training/validaiton/testing set')
    parser.add_argument('-Ntot', '--Ntotal', type=int, default=10,
                        help='number of available datasets with the same name as specified dataset')
    parser.add_argument('-Nval', '--Nvalidation', type=int, default=10000,
                        help='')
    parser.add_argument('-d', '--dataset', type=str,
                        help='your dataset')
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
        self.kernel_size = np.array(args.filter_size.split(",")).astype('int')
        self.stride = np.array(args.filter_stride.split(",")).astype('int')
        self.dilation = np.array(args.dilation.split(",")).astype('int')
        self.activation = np.array(args.activation_functions.split(','))
        self.dropout = np.array(args.dropout.split(",")).astype('float')
        self.pooling = np.array(args.pooling.split(',')).astype('bool')
        self.pool_size = np.array(args.pool_size.split(",")).astype('int')
        self.pool_stride = np.array(args.pool_stride.split(",")).astype('int')



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


def network(args, netargs, shape, outdir, data, targets):

    model = Sequential()

    optimizer = choose_optimizer(args)

    #TODO: add support for advanced activation functions

    count = 0
    for i, op in enumerate(netargs.features):

        if int(op) == 1:
            count+=1
            #print(count)
            # standard convolutional layer with max pooling
            model.add(Conv1D(
                netargs.Nfilters[i],
                input_shape=(512, 1),
                kernel_size=netargs.kernel_size[i],
                strides= netargs.stride[i],
                padding= 'valid',
                dilation_rate=netargs.dilation[i],
                use_bias=True,
                kernel_initializer=initializers.glorot_normal(),
                bias_initializer=initializers.glorot_normal(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
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
                print(netargs.pool_size[i])
                model.add(MaxPool1D(
                    pool_size=netargs.pool_size[i],
                    strides=None,
                    #strides=netargs.pool_stride[i],
                    padding='valid',
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
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", "binary_crossentropy"]
    )

    model.summary()

    #TODO: add options to enable/disable certain callbacks


    clr = CyclicLR(base_lr=args.lr, max_lr=args.max_learning_rate, step_size=args.stepsize)

    earlyStopping = EarlyStopping(monitor='val_acc', patience=args.patience, verbose=0, mode='auto')

    redLR = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=args.LRpatience, verbose=0, mode='auto',
                              epsilon=0.0001, cooldown=0, min_lr=0)

    modelCheck = ModelCheckpoint('{0}/best_weights.hdf5'.format(outdir), monitor='val_acc', verbose=0, save_best_only=True,save_weights_only=True, mode='auto', period=0)


    print('Fitting model...')
    if args.lr != args.max_learning_rate:
        hist = model.fit(data, targets,
                         epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         #class_weight=netargs.class_weight,
                         validation_split=0.10,
                         shuffle=True,
                         verbose=1,
                         callbacks=[clr, earlyStopping, redLR, modelCheck])
    else:
        hist = model.fit(data, targets,
                         epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         #class_weight=netargs.class_weight,
                         validation_split=0.10,
                         shuffle=True,
                         verbose=1,
                         callbacks=[earlyStopping, modelCheck])

    print('Evaluating model...')

    model.load_weights('{0}/best_weights.hdf5'.format(outdir))

    p = model.predict(data[len(data) - len(data) /5:])
    t = targets[len(data) - len(data) / 5:]

    #eval_results = model.evaluate(p, t,
    #                              sample_weight=None,
    #                              batch_size=args.batch_size, verbose=1)

    #preds = model.predict(p)

    gr, gt = (p[t==0] < 0.5).sum(), (t==0).sum()
    sr, st = (p[t==1] > 0.5).sum(), (t==1).sum()

    print i, "Glitch Accuracy: %1.3f" % (float(gr) / float(gt))
    print i, "Signal Accuracy: %1.3f" % (float(sr) / float(st))

    return model, hist, p



def main(args):
    # get arguments
    # convert args to correct format for network
    netargs = network_args(args)


    # read the samples in
    f = h5py.File(args.dataset, 'r')

    glitch = f['noise'][:]
    signal = f['signal'][:]

    data = np.concatenate([glitch, signal])
    targets = np.zeros(len(data))
    targets[:(len(targets)/2)] = 1

    #Randomize sample positions
    idx = np.arange(0, len(data), 1)
    np.random.shuffle(idx)
    data = data[idx]
    targets = targets[idx]

    if not os.path.exists('{0}'.format(args.outdir)):
        os.makedirs('{0}'.format(args.outdir))

    Nrun = 0
    while os.path.exists('{0}/run{1}'.format(args.outdir,Nrun)):
        Nrun += 1
    os.makedirs('{0}/run{1}'.format(args.outdir, Nrun))

    width = 512
    shape = width
    out = '{0}/run{1}'.format(args.outdir, Nrun)


    # train and test network
    model, hist, preds = network(args, netargs, shape, out,
                                               data, targets)


    with open('{0}/run{1}/args.pkl'.format(args.outdir, Nrun), "wb") as wfp:
        pickle.dump(args, wfp)

    for m in model.metrics_names:
        print('Test {0}:'.format(m))

    #shutil.copy('./runCNN.sh', '{0}/SNR{1}/run{2}'.format(args.outdir, args.SNR,Nrun))

    model.save('{0}/run{1}/nn_model.hdf5'.format(args.outdir,Nrun))
    np.save('{0}/run{1}/targets.npy'.format(args.outdir,Nrun),y_test)
    np.save('{0}/run{1}/preds.npy'.format(args.outdir,Nrun), preds)
    np.save('{0}/run{1}/history.npy'.format(args.outdir,Nrun), hist.history)
    #np.save('{0}/run{1}/test_results.npy'.format(args.outdir,Nrun),eval_results)

    print('Results saved at: {0}/run{1}'.format(args.outdir,Nrun))

if __name__ == '__main__':
    args = parser()
    main(args)
