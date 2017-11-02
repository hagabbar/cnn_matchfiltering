#!/bin/bash

# Use GPU:
# May need flags
#export THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float32,gpuarray.preallocate=0.9"
#export CPATH=$CPATH:/home/2136420/theanoenv/include
#export LIBRARY_PATH=$LIBRARY_PATH:/home/2136420/theanoenv/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/2136420/theanoenv/lib


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

# Location and name of training/validation/test sets:
# set for use on deimos
training_dataset=/home/chrism/deepdata_bbh/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_1seed_ts_0.sav
val_dataset=/home/chrism/deepdata_bbh/BBH_validation_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${3}_1seed_ts_0.sav
test_dataset=/home/chrism/deepdata_bbh/BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${3}_1seed_ts_0.sav


Nts=10000               # Number of time series
Nval=10000              # Number of time series for validation/testing

# Learning constraints:
learning_rate=0.01
max_learning_rate=0.01
stepsize=1000
momentum=0.9
n_epochs=30
batch_size=10
patience=10
LRpatience=5

###########################################
# 'operation_string' operations:
# o covolutional layer + max-pooling -> 1
# o fully connected layer -> 0
# o classification -> 4
###########################################

operation_string="1,1,1,1,0,4"
nkerns="8,16,32,64,32,2"
filter_size="1-32,1-16,1-8,1-4,0-0,0-0"
filter_stride="1-1,1-1,1-1,1-1"
dilation="1-1,1-1,1-1,1-1,1-1"
pooling="1,1,1,1"
pool_size="1-8,1-6,1-4,1-2"
pool_stride="1-8,1-6,1-4,1-2"
dropout="0.0,0.0,0.0,0.0,0.5,0.0"

functions="elu,elu,elu,elu,elu,softmax"

# update function to use
opt_fn="SGD"
momentum=0.9

#TODO: reformat args and add missing args

./CNN-keras.py -SNR=${1} -Nts=$Nts -Nval=$Nval -Trd=$training_dataset -Vald=$val_dataset -Tsd=$test_dataset -lr=$learning_rate -mlr=$max_learning_rate -ss=$stepsize -mn=$momentum -f=$operation_string -bs=$batch_size -nf=$nkerns -fs=$filter_size -fst=$filter_stride -fpd=$filter_pad -ps=$pool_size -pst=$pool_stride -ppd=$pool_pad -dp=$dropout -fn=$functions -pt=$patience -lpt=$LRpatience -dl=$dilation -p=$pooling
