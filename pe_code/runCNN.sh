#!/bin/bash

# Use GPU:
# May need flags
#export THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float32,gpuarray.preallocate=0.9"
#export CPATH=$CPATH:/home/2136420/theanoenv/include
#export LIBRARY_PATH=$LIBRARY_PATH:/home/2136420/theanoenv/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/2136420/theanoenv/lib

export CUDA_VISIBLE_DEVICES=1

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

# snr 10
seed_nums=("10" "20" "150")
# snr 8
#seed_nums=("8" "18" "208")

# Location and name of training/validation/test sets:
# For use on LHO
datapath=/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data
training_dataset=/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_$seed_nums[0]seed_ts_0.sav
val_dataset=/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${3}_$seed_nums[1]seed_ts_0.sav
test_dataset=/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${3}_$seed_nums[2]seed_ts_0.sav
training_params=/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_$seed_nums[0]seed_params_0.sav
test_params=/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${2}_$seed_nums[2]seed_params_0.sav
val_params=/home/hunter.gabbard/glasgow/github_repo_code/cnn_matchfiltering/data/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${2}_$seed_nums[1]seed_params_0.sav

# For use on LLO
#datapath=/home/hunter.gabbard/CBC/cnn_matchfiltering/data
#training_dataset=/home/hunter.gabbard/CBC/cnn_matchfiltering/data/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_8seed_ts_0.sav
#val_dataset=/home/hunter.gabbard/CBC/cnn_matchfiltering/data/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${3}_18seed_ts_0.sav
#test_dataset=/home/hunter.gabbard/CBC/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${3}_150seed_ts_0.sav
#training_params=/home/hunter.gabbard/CBC/cnn_matchfiltering/data/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_8seed_params_0.sav
#test_params=/home/hunter.gabbard/CBC/cnn_matchfiltering/data/BBH_testing_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${2}_150seed_params_0.sav
#val_params=/home/hunter.gabbard/CBC/cnn_matchfiltering/data/BBH_validation_1s_8192Hz_10Ksamp_1n_iSNR${1}_Hdet_${2}_18seed_params_0.sav


# For use on deimos
#datapath=/home/chrism/deepdata_bbh
#training_dataset=/home/chrism/deepdata_bbh/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_1seed_ts_0.sav
#val_dataset=/home/chrism/deepdata_bbh/BBH_validation_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${3}_1seed_ts_0.sav
#test_dataset=/home/chrism/deepdata_bbh/BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${3}_1seed_ts_0.sav
#training_params=/home/chrism/deepdata_bbh/BBH_training_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_1seed_params_0.sav
#test_params=/home/chrism/deepdata_bbh/BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_1seed_params_0.sav
#val_params=/home/chrism/deepdata_bbh/BBH_validation_1s_8192Hz_10Ksamp_25n_iSNR${1}_Hdet_${2}_1seed_params_0.sav

Nts=10000               # Number of time series
Nval=1000              # Number of time series for validation/testing
Ntot=10

# Learning constraints:
learning_rate=0.001
max_learning_rate=0.001
decay=0.0
stepsize=1000
momentum=0.9
n_epochs=60
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

# original classification network
features="1,1,1,1,1,1,1,1,0,4"
nkerns="8,16,16,32,64,64,128,128,256,1"
filter_size="1-32,1-16,1-16,1-16,1-8,1-8,1-4,1-4"
filter_stride="1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1"
dilation="1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1"
pooling="1,0,0,0,1,0,0,1"
pool_size="1-8,1-1,1-1,1-1,1-6,1-1,1-1,1-4"
pool_stride="1-8,1-1,1-1,1-1,1-6,1-1,1-1,1-4"
dropout="0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0"

functions="elu,elu,elu,elu,elu,elu,elu,elu,elu,linear"

# larger network
#features="1,1,1,1,1,1,1,1,1,1,0,4"
#nkerns="8,8,16,16,32,32,64,64,128,128,64,1"
#filter_size="1-32,1-32,1-16,1-16,1-16,1-16,1-8,1-8,1-4,1-4"
#filter_stride="1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1"
#dilation="1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1,1-1"
#pooling="1,0,0,0,0,0,1,0,0,1"
#pool_size="1-8,1-1,1-1,1-1,1-1,1-1,1-6,1-1,1-1,1-4"
#pool_stride="1-8,1-1,1-1,1-1,1-1,1-1,1-6,1-1,1-1,1-4"
#dropout="0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0"

#functions="elu,elu,elu,elu,elu,elu,elu,elu,elu,elu,elu,linear"


# reduced network
#features="1,1,1,1,1,0,4"
#nkerns="8,16,32,64,128,256,1"
#filter_size="1-32,1-16,1-8,1-4,1-2"
#filter_stride="1-1,1-1,1-1,1-1,1-1"
#dilation="1-1,1-1,1-1,1-1,1-1"
#pooling="1,1,1,1,1,0"
#pool_size="1-8,1-8,1-6,1-4,1-2"
#pool_stride="1-8,1-8,1-6,1-4,1-2"
#dropout="0.0,0.0,0.0,0.0,0.0,0.1,0.0"

#functions="elu,elu,elu,elu,elu,elu,linear"

./CNN-pe.py -SNR=${1} -Nts=$Nts -Ntot=$Ntot -Nval=$Nval \
 -Trd=$training_dataset -Vald=$val_dataset -Tsd=$test_dataset -bs=$batch_size\
 --training_dtype=${2} --testing_dtype=${3} --datapath=$datapath \
 -Trp=$training_params -Valp=$val_params -Tsp=$test_params \
 -opt=$opt -lr=$learning_rate -mlr=$max_learning_rate -NE=$n_epochs -dy=$decay \
 -ss=$stepsize -mn=$momentum --nesterov=$nesterov --rho=$rho --epsilon=$epsilon \
 --beta_1=$beta_1 --beta_2=$beta_2 -pt=$patience -lpt=$LRpatience \
 -f=$features  -nf=$nkerns -fs=$filter_size -fst=$filter_stride -fpd=$filter_pad \
 -dl=$dilation  -p=$pooling -ps=$pool_size -pst=$pool_stride -ppd=$pool_pad \
 -dp=$dropout -fn=$functions \
 -od=$outdir
