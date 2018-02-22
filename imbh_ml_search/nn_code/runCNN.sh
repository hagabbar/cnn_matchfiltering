#!/bin/bash

# Use GPU:
# May need flags
#export THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float32,gpuarray.preallocate=0.9"
#export CPATH=$CPATH:/home/2136420/theanoenv/include
#export LIBRARY_PATH=$LIBRARY_PATH:/home/2136420/theanoenv/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/2136420/theanoenv/lib

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

# Location and name of training/validation/test sets:
# set for use on deimos
dataset=/home/hunter.gabbard/CBC/imbh_ml_search/data/outsmall.hdf

Nts=100000               # Number of time series
Nval=10000              # Number of time series for validation/testing
Ntot=10

# Learning constraints:
learning_rate=0.001
max_learning_rate=0.005
decay=0.0
stepsize=32
momentum=0.9
n_epochs=200
batch_size=32
patience=10
LRpatience=5


outdir="./history"

# update function to use
opt="Nadam"
# parameters for particular optimizers
nesterov=true
rho=0.9
epsilon=0.000000001
beta_1=0.9
beta_2=0.999

###########################################
# 'features' operations:
# o covolutional layer + max-pooling -> 1
# o fully connected layer -> 0
# o classification -> 4
###########################################

features="1,1,1,1,1,1,1,0,4"
nkerns="8,16,32,64,64,128,128,64,1"
filter_size="32,16,16,16,8,8,4,4"
filter_stride="1,1,1,1,1,1,1,1"
dilation="1,1,1,1,1,1,1"
pooling="0,0,0,0,0,0,0"
pool_size="8,1,1,6,1,1,4"
pool_stride="8,1,1,6,1,1,4"
dropout="0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0"

functions="elu,elu,elu,elu,elu,elu,elu,elu,elu,sigmoid"

./CNN-keras.py -Nts=$Nts -Ntot=$Ntot -Nval=$Nval \
 -d=$dataset -bs=$batch_size\
 -opt=$opt -lr=$learning_rate -mlr=$max_learning_rate -NE=$n_epochs -dy=$decay \
 -ss=$stepsize -mn=$momentum --nesterov=$nesterov --rho=$rho --epsilon=$epsilon \
 --beta_1=$beta_1 --beta_2=$beta_2 -pt=$patience -lpt=$LRpatience \
 -f=$features  -nf=$nkerns -fs=$filter_size -fst=$filter_stride -fpd=$filter_pad \
 -dl=$dilation  -p=$pooling -ps=$pool_size -pst=$pool_stride -ppd=$pool_pad \
 -dp=$dropout -fn=$functions \
 -od=$outdir
