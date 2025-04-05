#!/bin/bash
#SBATCH -n 4                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-25:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p murphy           # Partition to submit to
#SBATCH --mem=50G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o jobs_out/may3/T=50_n=100_actionC=1_N=2k_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e jobs_out/may3/T=50_n=100_actionC=1_N=2k_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/2020.11
cd /n/home02/kellywzhang

source activate py39
which python

cd /n/home02/kellywzhang/adaptive-sandwich

# T=25,50
# Steepness=0.5,1,2?
# Sample size=100,500,1000

T=50
N=2000
recruit_t=1
decisions_between_updates=1
min_users=1
#synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_01_5_dosage'
#synthetic_mode='test_1_1_T2'
#synthetic_mode='delayed_effects_large'
eta=0 # Nowell: I think this is ignored
# RL_alg="posterior_sampling"
#RL_alg="fixed_randomization"
#err_corr='independent'
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
inference_mode="model" # This makes the inference model have three features: intercept, past_reward, and action
#inference_mode="value"
action_centering=1 # Nowell: This doesn't do anything if sigmoid LS used

recruit_n=50
n=50
# recruit_n=100
# n=100
# recruit_n=500
# n=500



save_dir=/n/murphy_lab/lab/kellywzhang/inference_after_pooling/results/may3

#for steepness in 1 2 0.5 5
for steepness in 0.5 1 5
do

    for synthetic_mode in 'delayed_1_dosage' 'delayed_5_dosage' #'delayed_2_dosage'
    do
        python RL_Study_Simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --save_dir=$save_dir --action_centering $action_centering

        python After_Study_Analyses.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --debug 0 --RL_alg $RL_alg --eta $eta --alg_state_feats $alg_state_feats --inference_mode $inference_mode --save_dir=$save_dir --action_centering $action_centering
    done

done
