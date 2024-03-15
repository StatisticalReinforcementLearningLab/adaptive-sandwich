#!/bin/bash
#SBATCH -n 4                                      # Number of cores
#SBATCH -N 1                                      # Ensure that all cores are on one machine
#SBATCH -t 0-0:15                                 # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=20G                                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p gpu_requeue                            # Target Partition
#SBATCH --gres=gpu:1                              # Request a GPU
#SBATCH -o slurm.%N.%A.%a.out                     # STDOUT
#SBATCH -e slurm.%N.%A.%a.err                     # STDERR
#SBATCH --mail-type=END                           # This command would send an email when the job ends.
#SBATCH --mail-type=FAIL                          # This command would send an email when the job ends.
#SBATCH --mail-user=nowellclosser@g.harvard.edu   # Email to which notifications will be sent

# Note this script is to be run with something like the following command:
# sbatch --array=[0-99] simulation_run_parallel.sh --T=25 --n=100 --recruit_n=100 --recruit_t=1

# To analyze, run simulation_collect_analyses.sh as described in the
# output of one of the simulation runs.

# Stop on nonzero exit codes and use of undefined variables, and print all commands
set -eu

echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Parsing options.

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# Defaults. Overridden if supplied.
T=3
N=100
recruit_t=1
decisions_between_updates=1
recruit_n=100
n=100
min_users=1
RL_alg="sigmoid_LS"
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
action_centering=1
steepness=0.0  # Note that the ".0" is critical because of the way the filenames below are formed
synthetic_mode="delayed_1_dosage"

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
# Note that the N argument is not supplied here: the number of simulations is
# determined by the number of jobs in the slurm job array.
while getopts T:t:n:u:d:m:r:e:f:a:s:y:l:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    T  | max_time )                     needs_arg; T="$OPTARG" ;;
    t  | recruit_t )                    needs_arg; recruit_t="$OPTARG" ;;
    n  | num_users )                    needs_arg; n="$OPTARG" ;;
    u  | recruit_n )                    needs_arg; recruit_n="$OPTARG" ;;
    d  | decisions_between_updates )    needs_arg; decisions_between_updates="$OPTARG" ;;
    m  | min_users )                    needs_arg; min_users="$OPTARG" ;;
    r  | RL_alg )                       needs_arg; RL_alg="$OPTARG" ;;
    e  | err_corr )                     needs_arg; err_corr="$OPTARG" ;;
    f  | alg_state_feats )              needs_arg; alg_state_feats="$OPTARG" ;;
    a  | action_centering )             needs_arg; action_centering="$OPTARG" ;;
    s  | steepness )                    needs_arg; steepness="$OPTARG" ;;
    y  | synthetic_mode )               needs_arg; synthetic_mode="$OPTARG" ;;
    \? )                                exit 2 ;;  # bad short option (error reported via getopts)
    * )                                 die "Illegal option --$OPT" ;; # bad long option
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Load Python 3.10, among other things
echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Loading mamba and CUDA modules.
module load Mambaforge/22.11.1-fasrc01
module load cuda/12.2.0-fasrc01

# Make virtualenv if necessary, and then activate it
cd ~
if ! test -d venv; then
  echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Creating venv, as it did not exist.
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd ~/adaptive-sandwich
echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Making sure Python requirements are installed.
pip install -r simulation_requirements.txt
echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: All Python requirements installed.

save_dir_prefix="/n/murphy_lab/lab/nclosser/adaptive_sandwich_simulation_results/${SLURM_ARRAY_JOB_ID}"

if test -d save_dir_prefix; then
  die 'Output directory already exists. Please supply a unique label, perhaps a datetime.'
fi
save_dir="${save_dir_prefix}/${SLURM_ARRAY_TASK_ID}"
save_dir_glob="${save_dir_prefix}/*"
mkdir -p "$save_dir"

# Simulate an RL study with the supplied arguments.  (We do just one repetition)
echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Beginning RL simulations.
python rl_study_simulation.py --T=$T --N=1 --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering --save_dir=$save_dir
echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Finished RL simulations.

# Create a convenience variable that holds the output folder for the last script
output_folder="${save_dir}/simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering}"
output_folder_glob="${save_dir_glob}/simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering}"

# Loop through each dataset created in the simulation (determined by number of Monte carlo repetitions)
# and do after-study analysis
echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Beginning after-study analysis.
python after_study_analysis.py analyze-dataset --study_dataframe_pickle="${output_folder}/exp=1/study_df.pkl" --rl_algorithm_object_pickle="${output_folder}/exp=1/study_RLalg.pkl"
echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Finished after-study analysis.

echo $(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: Simulation complete.
echo "$(date +"%Y-%m-%d %T") simulation_run_and_analysis_parallel.sh: When all jobs have completed, you may collect and summarize the analyses with ./simulation_collect_analyses.sh --input_glob=${output_folder_glob}/exp=1/analysis.pkl"
