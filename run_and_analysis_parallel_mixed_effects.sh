#!/bin/bash
#SBATCH -n 4                                                                                                # Number of cores
#SBATCH -N 1                                                                                                # Ensure that all cores are on one machine
#SBATCH -t 0-0:20                                                                                           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=5G                                                                                            # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p serial_requeue                                                                                   # Target Partition
#SBATCH -o /n/netscratch/murphy_lab/Lab/nclosser/adaptive_sandwich_simulation_results/%A/slurm.%a.out       # STDOUT
#SBATCH -e /n/netscratch/murphy_lab/Lab/nclosser/adaptive_sandwich_simulation_results/%A/slurm.%a.out       # STDERR
#SBATCH --mail-type=END                                                                                     # This command would send an email when the job ends.
#SBATCH --mail-type=FAIL                                                                                    # This command would send an email when the job ends.
#SBATCH --mail-user=nowellclosser@g.harvard.edu                                                             # Email to which notifications will be sent

# Note this script is to be run with something like the following command:
# sbatch --array=[0-99] run_and_analysis_parallel_mixed_effects.sh --T=25 --n=100 --recruit_n=100 --recruit_t=1

# To analyze, run simulation_collect_analyses.sh as described in the
# output of one of the simulation runs.

# If running on GPU, the following can be used:
# S BATCH -p gpu_requeue                                                       # Target Partition
# S BATCH --gres=gpu:1                                                         # Request a GPU

# Stop on nonzero exit codes and use of undefined variables
set -eu

echo "$(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects: Beginning simulation."

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# Arguments that only affect simulation side.
num_users=100
num_time_steps=10
delta_seed=0
beta_mean=1
beta_std=1
gamma_std=.1
sigma_e2=.1
policy_type="mixed_effects"
only_analysis=0

# Arguments that only affect inference side.
in_study_col_name="in_study_indicator"
action_col_name="action"
policy_num_col_name="policy_number"
calendar_t_col_name="calendar_time"
user_id_col_name="user_id"
action_prob_col_name="action_probability"
action_prob_func_filename="functions_to_pass_to_analysis/mixed_effects_action_selection.py"
action_prob_func_args_beta_index=0
alg_update_func_filename="functions_to_pass_to_analysis/mixed_effects_RL_estimating_function.py"
alg_update_func_type="estimating"
alg_update_func_args_beta_index=0
alg_update_func_args_action_prob_index=-1
alg_update_func_args_action_prob_times_index=-1
inference_func_filename="functions_to_pass_to_analysis/mixed_effects_primary_analysis_loss.py"
inference_func_args_theta_index=0
inference_func_type="loss"
theta_calculation_func_filename="functions_to_pass_to_analysis/mixed_effects_estimate_theta_primary_analysis.py"
suppress_interactive_data_checks=1
suppress_all_data_checks=0
small_sample_correction="none"
adaptive_bread_inverse_stabilization_method="trim_small_singular_values"

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts m:T:s:S:G:t:g:e:O:o:i:c:p:C:U:E:P:b:l:Z:B:D:j:I:h:J:H:Q:q:z:w:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    m  | num_users )                                    needs_arg; num_users="$OPTARG" ;;
    T  | num_time_steps )                               needs_arg; num_time_steps="$OPTARG" ;;
    s  | seed )                                         needs_arg; seed="$OPTARG" ;;
    S  | delta_seed )                                   needs_arg; delta_seed="$OPTARG" ;;
    G  | beta_mean )                                    needs_arg; beta_mean="$OPTARG" ;;
    t  | beta_std )                                     needs_arg; beta_std="$OPTARG" ;;
    g  | gamma_std )                                    needs_arg; gamma_std="$OPTARG" ;;
    e  | sigma_e2 )                                     needs_arg; sigma_e2="$OPTARG" ;;
    O  | policy_type )                                  needs_arg; policy_type="$OPTARG" ;;
    o  | only_analysis )                                needs_arg; only_analysis="$OPTARG" ;;
    i  | in_study_col_name )                            needs_arg; in_study_col_name="$OPTARG" ;;
    c  | action_col_name )                              needs_arg; action_col_name="$OPTARG" ;;
    p  | policy_num_col_name )                          needs_arg; policy_num_col_name="$OPTARG" ;;
    C  | calendar_t_col_name )                          needs_arg; calendar_t_col_name="$OPTARG" ;;
    U  | user_id_col_name )                             needs_arg; user_id_col_name="$OPTARG" ;;
    E  | action_prob_col_name )                         needs_arg; action_prob_col_name="$OPTARG" ;;
    P  | action_prob_func_filename )                    needs_arg; action_prob_func_filename="$OPTARG" ;;
    b  | action_prob_func_args_beta_index )             needs_arg; action_prob_func_args_beta_index="$OPTARG" ;;
    l  | alg_update_func_filename )                     needs_arg; alg_update_func_filename="$OPTARG" ;;
    Z  | alg_update_func_type )                         needs_arg; alg_update_func_type="$OPTARG" ;;
    B  | alg_update_func_args_beta_index )              needs_arg; alg_update_func_args_beta_index="$OPTARG" ;;
    D  | alg_update_func_args_action_prob_index )       needs_arg; alg_update_func_args_action_prob_index="$OPTARG" ;;
    j  | alg_update_func_args_action_prob_times_index ) needs_arg; alg_update_func_args_action_prob_times_index="$OPTARG" ;;
    I  | inference_func_filename )                      needs_arg; inference_func_filename="$OPTARG" ;;
    h  | inference_func_args_theta_index )              needs_arg; inference_func_args_theta_index="$OPTARG" ;;
    J  | inference_func_type )                          needs_arg; inference_func_type="$OPTARG" ;;
    H  | theta_calculation_func_filename )              needs_arg; theta_calculation_func_filename="$OPTARG" ;;
    Q  | suppress_interactive_data_checks )             needs_arg; suppress_interactive_data_checks="$OPTARG" ;;
    q  | suppress_all_data_checks )                     needs_arg; suppress_all_data_checks="$OPTARG" ;;
    z  | small_sample_correction )                      needs_arg; small_sample_correction="$OPTARG" ;;
    w  | adaptive_bread_inverse_stabilization_method )  needs_arg; adaptive_bread_inverse_stabilization_method="$OPTARG" ;;
    \? )                                        exit 2 ;;  # bad short option (error reported via getopts)
    * )                                         die "Illegal option --$OPT" ;; # bad long option
  esac
done

shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Check for invalid options that do not start with a dash. This
# prevents accidentally missing dashes and thinking you passed an
# arg that you didn't.
for arg in "$@"; do
  if [[ "$arg" != -* ]]; then
    die "Invalid argument: $arg. Options must start with a dash (- or --)."
  fi
done

# Load Python 3.10, among other things
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects.sh: Loading mamba and CUDA modules.
module load Mambaforge/22.11.1-fasrc01
# if using GPU, something like the following will be necessary:
# module load cuda/12.2.0-fasrc01

# Make virtualenv if necessary, and then activate it
cd ~
if ! test -d venv; then
  echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects.sh: Creating venv, as it did not exist.
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd ~/adaptive-sandwich
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects.sh: Making sure Python requirements are installed.
pip install -r requirements.txt
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects.sh: All Python requirements installed.

save_dir_prefix="/n/netscratch/murphy_lab/Lab/nclosser/adaptive_sandwich_simulation_results/${SLURM_ARRAY_JOB_ID}"

if test -d save_dir_prefix; then
  die 'Output directory already exists. Please supply a unique label, perhaps a datetime.'
fi
save_dir="${save_dir_prefix}/${SLURM_ARRAY_TASK_ID}"
save_dir_glob="${save_dir_prefix}/*"
mkdir -p "$save_dir"

# Simulate an miwaves RL study (unless we just want to analyze previous results)
if [ "$only_analysis" -eq "0" ]; then
  echo "$(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects: Beginning RL study simulation."
  python mixed_effects_sample_data/src/run_simulation.py \
    --num_users $num_users \
    --num_time_steps $num_time_steps \
    --seed $SLURM_ARRAY_TASK_ID \
    --delta_seed $delta_seed \
    --beta_mean $beta_mean \
    --beta_std $beta_std \
    --gamma_std $gamma_std \
    --sigma_e2 $sigma_e2 \
    --policy_type $policy_type \
    --save_dir $save_dir
  echo "$(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects: Finished RL study simulation."
fi

# Create a convenience variable that holds the output folder for the last script
output_folder="${save_dir}"
output_folder_glob="${save_dir_glob}"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects: Beginning after-study analysis."
python after_study_analysis.py analyze-dataset \
  --study_df_pickle="${output_folder}/study_df.pkl" \
  --action_prob_func_filename=$action_prob_func_filename \
  --action_prob_func_args_pickle="${output_folder}/action_selection_function_dict.pkl" \
  --action_prob_func_args_beta_index=$action_prob_func_args_beta_index \
  --alg_update_func_filename=$alg_update_func_filename \
  --alg_update_func_type=$alg_update_func_type \
  --alg_update_func_args_pickle="${output_folder}/estimating_equation_function_dict.pkl" \
  --alg_update_func_args_beta_index=$alg_update_func_args_beta_index \
  --alg_update_func_args_action_prob_index=$alg_update_func_args_action_prob_index \
  --alg_update_func_args_action_prob_times_index=$alg_update_func_args_action_prob_times_index \
  --inference_func_filename=$inference_func_filename \
  --inference_func_args_theta_index=$inference_func_args_theta_index \
  --inference_func_type=$inference_func_type \
  --theta_calculation_func_filename=$theta_calculation_func_filename \
  --in_study_col_name=$in_study_col_name \
  --action_col_name=$action_col_name \
  --policy_num_col_name=$policy_num_col_name \
  --calendar_t_col_name=$calendar_t_col_name \
  --user_id_col_name=$user_id_col_name \
  --action_prob_col_name=$action_prob_col_name \
  --suppress_interactive_data_checks=$suppress_interactive_data_checks \
  --suppress_all_data_checks=$suppress_all_data_checks \
  --small_sample_correction=$small_sample_correction \
  --adaptive_bread_inverse_stabilization_method=$adaptive_bread_inverse_stabilization_method
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects.sh: Finished after-study analysis.

echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects.sh: Simulation complete.
echo "$(date +"%Y-%m-%d %T") run_and_analysis_parallel_mixed_effects.sh: When all jobs have completed, you may collect and summarize the analyses with: bash simulation_collect_analyses.sh --input_glob=${output_folder_glob}/analysis.pkl --num_users=$num_users --index_to_check_ci_coverage=1  --in_study_col_name=$in_study_col_name --action_col_name=$action_col_name --action_prob_col_name=$action_prob_col_name
