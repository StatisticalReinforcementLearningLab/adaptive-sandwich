#!/bin/bash
set -e
set -x

#SBATCH -n 4                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-25:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p murphy           # Partition to submit to
#SBATCH --mem=50G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o jobs_out/may3/T=50_n=100_actionC=1_N=2k_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e jobs_out/may3/T=50_n=100_actionC=1_N=2k_%j.err  # File to which STDERR will be written, %j inserts jobid

# module load Anaconda3/2020.11
# cd /n/home02/kellywzhang

# source activate py39
# which python

# cd /n/home02/kellywzhang/adaptive-sandwich

# T=25,50
# Steepness=0.5,1,2?
# Sample size=100,500,1000

T=10
N=2
recruit_t=1
decisions_between_updates=2
min_users=1
#synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_01_5_dosage'
#synthetic_mode='test_1_1_T2'
#synthetic_mode='delayed_effects_large'
eta=0
RL_alg="sigmoid_LS"
# RL_alg="posterior_sampling"
#RL_alg="fixed_randomization"
#err_corr='independent'
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
inference_mode="model"
#inference_mode="value"
action_centering=1

recruit_n=100
n=100
#synthetic_mode='delayed_1_dosage'
#steepness=5

#n = 1000, 100, 500
#steepness = 1 2 0.5

#TODO: use this
save_dir=/n/murphy_lab/lab/kellywzhang/inference_after_pooling/results/may3

#for steepness in 1 2 0.5 5
for steepness in 1.0 0.5 5.0
do

    for synthetic_mode in 'delayed_1_dosage' 'delayed_5_dosage' #'delayed_2_dosage'
    do
        # Simulate an RL study with the supplied arguments.  (We do just one repetition)
        python rl_study_simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering

        # Create a convenience variable that holds the output folder for the last script
        output_folder="simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering}"

        # Loop through each dataset created in the simulation (determined by number of Monte carlo repetitions)
        # and do after-study analysis
        python after_study_analysis.py analyze-multiple-datasets-and-compare-to-empirical-variance --input_folder="${output_folder}" --study_dataframe_pickle_filename="study_df.pkl" --rl_algorithm_object_pickle_filename="study_RLalg.pkl"
    done

done
