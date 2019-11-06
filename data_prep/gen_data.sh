#! /bin/bash

# script to generate BBH data for use in the prl 

snr=(7 6 8 5 9 4 10 3 11 2 12 1)
fs=8192
T=1
Nnoise=25
seed=1
outdir=/home/chrism/deepdata_bbh
Nb=10000
N_str='10K'

NT=100000
Nv=20000
Nt=20000

# loop over the different SNR values
for s in "${snr[@]}"
do

    # generate data for metric mass distributions
    # training
    outfile=${outdir}/BBH_training_${T}s_${fs}Hz_${N_str}samp_${Nnoise}n_iSNR${s}_Hdet_metricmass_${seed}seed
    python data_prep_bbh.py -I H1 -s ${s} -f ${fs} -z 1 -T ${T} -N ${NT} -Nn ${Nnoise} -Nb ${Nb} -m metric -b ${outfile}
    # validation
    outfile=${outdir}/BBH_validation_${T}s_${fs}Hz_${N_str}samp_${Nnoise}n_iSNR${s}_Hdet_metricmass_${seed}seed
    python data_prep_bbh.py -I H1 -s ${s} -f ${fs} -z 2 -T ${T} -N ${Nv} -Nn 1 -Nb ${Nb} -m metric -b ${outfile}
    # testing
    outfile=${outdir}/BBH_testing_${T}s_${fs}Hz_${N_str}samp_${Nnoise}n_iSNR${s}_Hdet_metricmass_${seed}seed
    python data_prep_bbh.py -I H1 -s ${s} -f ${fs} -z 3 -T ${T} -N ${Nt} -Nn 1 -Nb ${Nb} -m metric -b ${outfile}
    
    # generate data for astro mass distributions
    # training
    outfile=${outdir}/BBH_training_${T}s_${fs}Hz_${N_str}samp_${Nnoise}n_iSNR${s}_Hdet_astromass_${seed}seed
    python data_prep_bbh.py -I H1 -s ${s} -f ${fs} -z 4 -T ${T} -N ${NT} -Nn ${Nnoise} -Nb ${Nb} -m astro -b ${outfile}
    # validation
    outfile=${outdir}/BBH_validation_${T}s_${fs}Hz_${N_str}samp_${Nnoise}n_iSNR${s}_Hdet_astromass_${seed}seed
    python data_prep_bbh.py -I H1 -s ${s} -f ${fs} -z 5 -T ${T} -N ${Nv} -Nn 1 -Nb ${Nb} -m astro -b ${outfile}
    # testing
    outfile=${outdir}/BBH_testing_${T}s_${fs}Hz_${N_str}samp_${Nnoise}n_iSNR${s}_Hdet_astromass_${seed}seed
    python data_prep_bbh.py -I H1 -s ${s} -f ${fs} -z 6 -T ${T} -N ${Nt} -Nn 1 -Nb ${Nb} -m astro -b ${outfile}

done

