#!/bin/bash
#SBATCH -n 4                                      # Number of cores
#SBATCH -N 1                                      # Ensure that all cores are on one machine
#SBATCH -t 0-10:00                                 # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                         # Partition to submit to
#SBATCH --mem=20G                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm.%N.%j.out                        # STDOUT
#SBATCH -e slurm.%N.%j.err                        # STDERR
#SBATCH --mail-type=END                           # This command would send an email when the job ends.
#SBATCH --mail-type=FAIL                          # This command would send an email when the job ends.
#SBATCH --mail-user=nowellclosser@g.harvard.edu   # Email to which notifications will be sent

# Stop on nonzero exit codes and use of undefined variables, and print all commands
set -eu

echo $(date +"%Y-%m-%d %T") simulation_run.sh: Beginning simulation.

# Load Python 3.10, among other things
echo $(date +"%Y-%m-%d %T") simulation_run.sh: Loading mamba and CUDA modules.
module load Mambaforge/22.11.1-fasrc01
module load cuda/12.2.0-fasrc01

# Make virtualenv if necessary, and then activate it
cd ~
if ! test -d venv; then
  echo $(date +"%Y-%m-%d %T") simulation_run.sh: Creating venv, as it did not exist.
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd ~/adaptive-sandwich
echo $(date +"%Y-%m-%d %T") simulation_run.sh: Making sure Python requirements are installed.
pip install -r simulation_requirements.txt
echo $(date +"%Y-%m-%d %T") simulation_run.sh: All Python requirements installed.

# T=25,50
# Steepness=0.5,1,2?
# Sample size=100,500,1000

#TODO: Remove single-letter parameters in favor of descriptive names
T=3
N=100
recruit_t=1
decisions_between_updates=1
min_users=1
#TODO: All commented out options should probably just be in the python script documentation
#synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_01_5_dosage'
#synthetic_mode='test_1_1_T2'
#synthetic_mode='delayed_effects_large'
RL_alg="sigmoid_LS"
# RL_alg="posterior_sampling"
#RL_alg="fixed_randomization"
#err_corr='independent'
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
action_centering=1

recruit_n=100
n=100
#synthetic_mode='delayed_1_dosage'
#steepness=5

#n = 1000, 100, 500
#steepness = 1 2 0.5

#TODO: use this in after study analysis. Actually shouldn't need? Just put alongside inputs
now=$(printf "%(%F_%H-%M-%S)T")
save_dir="/n/murphy_lab/lab/nclosser/adaptive_sandwich_simulation_results/${now}"
mkdir -p "$save_dir"

# Note that the ".0" is critical because of the way the filenames below are formed
for steepness in 0.0
do
    for synthetic_mode in 'delayed_1_dosage'
    do
        # Simulate an RL study with the supplied arguments.  (We do just one repetition)
        echo $(date +"%Y-%m-%d %T") simulation_run.sh: Beginning RL simulation for steepness $steepness and synthetic_mode $synthetic_mode.
        python rl_study_simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering --save_dir=$save_dir
        echo $(date +"%Y-%m-%d %T") simulation_run.sh: Finished RL simulation.

        # Create a convenience variable that holds the output folder for the last script
        output_folder="${save_dir}/simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering}"

        # Loop through each dataset created in the simulation (determined by number of Monte carlo repetitions)
        # and do after-study analysis
        echo $(date +"%Y-%m-%d %T") simulation_run.sh: Beginning after-study analysis.
        python after_study_analysis.py analyze-multiple-datasets-and-compare-to-empirical-variance --input_folder="${output_folder}" --study_dataframe_pickle_filename="study_df.pkl" --rl_algorithm_object_pickle_filename="study_RLalg.pkl"
        echo $(date +"%Y-%m-%d %T") simulation_run.sh: Finished after-study analysis.
    done
done

echo $(date +"%Y-%m-%d %T") simulation_run.sh: Simulation complete.
