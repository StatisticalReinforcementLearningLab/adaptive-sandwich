#!/bin/bash
#SBATCH -n 4                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-25:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p murphy           # Partition to submit to
#SBATCH --mem=50G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=END                                    #This command would send an email when the job ends.
#SBATCH --mail-type=FAIL                                   #This command would send an email when the job ends.
#SBATCH --mail-user=nowellclosser@g.harvard.edu            #Email to which notifications will be sent

# Stop on nonzero exit codes and use of undefined variables
set -eu

# Load Python 3.10, among other things
module load Mambaforge/22.11.1-fasrc01

cd ~

# Make virtualenv if necessary, and then activate it
if test -d venv; then
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all requirements.  This is incremental, so it's ok to do every time.
cd ~/adaptive-sandwich
pip install --no-index -r requirements.txt

# T=25,50
# Steepness=0.5,1,2?
# Sample size=100,500,1000

T=10
N=2
recruit_t=1
decisions_between_updates=2
min_users=1
synthetic_mode='delayed_1_dosage'
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

#TODO: use this in after study analysis
now = $(printf "%(%F_%H%M%S)T")
save_dir="/n/murphy_lab/lab/nclosser/adaptive_sandwich_simulation_results/${now}"
mkdir -p $save_dir

#for steepness in 1 2 0.5 5
for steepness in 1.0 0.5 5.0
do

    for synthetic_mode in 'delayed_1_dosage' 'delayed_5_dosage' #'delayed_2_dosage'
    do
        # Simulate an RL study with the supplied arguments.  (We do just one repetition)
        python rl_study_simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering --save_dir=$save_dir

        # Create a convenience variable that holds the output folder for the last script
        output_folder="${save_dir}/simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering}"

        # Loop through each dataset created in the simulation (determined by number of Monte carlo repetitions)
        # and do after-study analysis
        python after_study_analysis.py analyze-multiple-datasets-and-compare-to-empirical-variance --input_folder="${output_folder}" --study_dataframe_pickle_filename="study_df.pkl" --rl_algorithm_object_pickle_filename="study_RLalg.pkl" --save_dir=$save_dir
    done

done
