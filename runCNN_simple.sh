#!/bin/bash

# specify GPU for Tensorflow
export CUDA_VISIBLE_DEVICES=0

#####################
# Running the network
#####################

# To run network call this file followed by snr and mass dist:
# ./runCNN.sh <snr> <training mass dist> <val/test mass dist>
# eg:
# ./runCNN.sh 8 metricmass metricmass
# options for mass dist are currently:
# - metricmass
# - astromass

#######################
# Data files and output
#######################

# sampling frequency
fs=1024

# Location and name of training/validation/test sets:
training_dataset=/home/michael/datasets/bbh/BBH_training_1s_${fs}Hz_10Ksamp_1n_iSNR${1}_Hdet_${2}_1seed_ts_0.pkl
val_dataset=/home/michael/datasets/bbh/BBH_validation_1s_${fs}Hz_10Ksamp_1n_iSNR${1}_Hdet_${3}_1seed_ts_0.pkl
test_dataset=/home/michael/datasets/bbh/BBH_testing_1s_${fs}Hz_10Ksamp_1n_iSNR${1}_Hdet_${3}_1seed_ts_0.pkl

Nts=10000                  # Number of time series
Nval=2000                  # Number of time series for validation/testing
Ntot=1                     # Number of files available

outdir="./history"         # Output directory

############################
# Learning rate and optimiser
############################

batch_size=100             # Number of samples to pass at once
n_epochs=200               # Number of epochs to train for

# Learning constraints:
learning_rate=0.001        # Base learning rate
max_learning_rate=0.001    # Maximum learning rate for cyclic learning rate (equal diasbles CLR)
decay=0.0                  # Learning rate decay
stepsize=1000              # Stepsize for cyclic learning rate
patience=20                # Early stopping (stop after n epochs with no improvemnt of validation)
LRpatience=1000            # Update the LR every n epochs

# Optimiser
opt="Adam"                 # Type
# Optimiser specific parameters (not always used)
momentum=0.9               # Momentum
nesterov=true              # Use Nesterov momentum (when applicable)
# More opitmiser parameters (see Keras docs for details)
rho=0.9
epsilon=0.000000001
beta_1=0.9
beta_2=0.999

###########################################
# Network config
###########################################
# 'features' operations:
# o covolutional layer + max-pooling -> 1
# o fully connected layer (with flattening)-> 0
# o fully connected layer -> 2
# o classification -> 4
###########################################

# Layer types (follows the above scheme)
features="1,1,0,2,4"
nkerns="16,16,16,16,2"
# Convolution options
filter_size="1-16,1-16"
filter_stride="1-1,1-1"
dilation="1-1,1-1"
# Max pooling options
pooling="1,1"
pool_size="1-4,1-4"
pool_stride="1-4,1-4"
# Dropout
dropout="0.0,0.0,0.0,0.0,0.0"
# Activation funcrtions
functions="elu,elu,elu,elu,softmax"

# Run everything
./CNN-keras.py -SNR=8 -Nts=$Nts -Ntot=$Ntot -Nval=$Nval \
 -Trd=$training_dataset -Vald=$val_dataset -Tsd=$test_dataset -bs=$batch_size\
 -opt=$opt -lr=$learning_rate -mlr=$max_learning_rate -NE=$n_epochs -dy=$decay \
 -ss=$stepsize -mn=$momentum --nesterov=$nesterov --rho=$rho --epsilon=$epsilon \
 --beta_1=$beta_1 --beta_2=$beta_2 -pt=$patience -lpt=$LRpatience \
 -f=$features  -nf=$nkerns -fs=$filter_size -fst=$filter_stride -fpd=$filter_pad \
 -dl=$dilation  -p=$pooling -ps=$pool_size -pst=$pool_stride -ppd=$pool_pad \
 -dp=$dropout -fn=$functions \
 -od=$outdir
