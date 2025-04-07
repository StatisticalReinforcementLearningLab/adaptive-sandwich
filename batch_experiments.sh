#!/bin/bash
#SBATCH -n 4                                                                             # Number of cores
#SBATCH -N 1                                                                             # Ensure that all cores are on one machine
#SBATCH -t 2-0:00                                                                       # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                                                                # Partition to submit to
#SBATCH --mem=100G                                                                       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/netscratch/murphy_lab/Lab/nclosser/kelly_paper_reproduction_output/%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/netscratch/murphy_lab/Lab/nclosser/kelly_paper_reproduction_output/%j.out  # File to which STDERR will be written, %j inserts jobid

set -eu

module load Mambaforge/22.11.1-fasrc01
cd /n/murphy_lab/lab/nclosser/kelly_paper_reproduction

# Load Python 3.10, among other things
module load Mambaforge/22.11.1-fasrc01
# module load cuda/12.2.0-fasrc01

# Make virtualenv if necessary, and then activate it
if ! test -d venv; then
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd code
pip install -r requirements.txt

T=50
N=2000
recruit_t=1
decisions_between_updates=1
min_users=1
#synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_01_5_dosage'
#synthetic_mode='test_1_1_T2'
#synthetic_mode='delayed_effects_large'
# RL_alg="posterior_sampling"
#RL_alg="fixed_randomization"
#err_corr='independent'
RL_alg="sigmoid_LS"
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
inference_mode="model" # Nowell: This makes the inference model have three features: intercept, past_reward, and action
#inference_mode="value"

action_centering=0 # Nowell: This doesn't do anything if sigmoid LS used
eta=0 # Nowell: This is ignored


save_dir=/n/murphy_lab/lab/nclosser/kelly_paper_reproduction/results/$SLURM_JOB_ID
mkdir -p $save_dir
for n in 50 100 500
do
    for steepness in 0.5 1 5
    do
        for synthetic_mode in 'delayed_1_dosage' 'delayed_5_dosage' 'delayed_1_dosage_paper' 'delayed_5_dosage_paper'
        do
            recruit_n=$n
            python RL_Study_Simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --save_dir=$save_dir --action_centering $action_centering

            python After_Study_Analyses.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --debug 0 --RL_alg $RL_alg --eta $eta --alg_state_feats $alg_state_feats --inference_mode $inference_mode --save_dir=$save_dir --action_centering $action_centering
        done

    done
done
