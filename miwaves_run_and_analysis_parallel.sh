#!/bin/bash
#SBATCH -n 4                                                                                                # Number of cores
#SBATCH -N 1                                                                                                # Ensure that all cores are on one machine
#SBATCH -t 0-0:20                                                                                           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=5G                                                                                            # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -p serial_requeue                                                                                   # Target Partition
#SBATCH -o /n/holyscratch01/murphy_lab/Lab/nclosser/adaptive_sandwich_simulation_results/%A/slurm.%a.out    # STDOUT
#SBATCH -e /n/holyscratch01/murphy_lab/Lab/nclosser/adaptive_sandwich_simulation_results/%A/slurm.%a.out    # STDERR
#SBATCH --mail-type=END                                                                                     # This command would send an email when the job ends.
#SBATCH --mail-type=FAIL                                                                                    # This command would send an email when the job ends.
#SBATCH --mail-user=nowellclosser@g.harvard.edu                                                             # Email to which notifications will be sent

# Note this script is to be run with something like the following command:
# sbatch --array=[0-99] miwaves_run_and_analysis_parallel.sh --T=25 --n=100 --recruit_n=100 --recruit_t=1

# To analyze, run simulation_collect_analyses.sh as described in the
# output of one of the simulation runs.

# If running on GPU, the following can be used:
# S BATCH -p gpu_requeue                                                       # Target Partition
# S BATCH --gres=gpu:1                                                         # Request a GPU

set -eu

echo "$(date +"%Y-%m-%d %T") miwaves_run_and_analysis_parallel: Beginning simulation."

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# Arguments that only affect simulation side.
seed=0
only_analysis=0

# Arguments that only affect inference side.
in_study_col_name="in_study_indicator"
action_col_name="action"
policy_num_col_name="policy_number"
calendar_t_col_name="calendar_time"
user_id_col_name="user_id"
action_prob_col_name="action_probability"
action_prob_func_filename="functions_to_pass_to_analysis/miwaves_action_selection.py"
action_prob_func_args_beta_index=0
rl_update_func_filename="functions_to_pass_to_analysis/miwaves_RL_estimating_function.py"
rl_update_func_type="estimating"
rl_update_func_args_beta_index=0
rl_update_func_args_action_prob_index=-1
rl_update_func_args_action_prob_times_index=-1
inference_loss_func_filename="functions_to_pass_to_analysis/miwaves_primary_analysis_loss.py"
inference_loss_func_args_theta_index=0
theta_calculation_func_filename="functions_to_pass_to_analysis/miwaves_estimate_theta_primary_analysis.py"
suppress_interactive_data_checks=0
suppress_all_data_checks=0
small_sample_correction="none"

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts i:c:p:C:U:E:P:b:l:Z:B:D:j:I:h:H:s:o:Q:q:z:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    i  | in_study_col_name )                            needs_arg; in_study_col_name="$OPTARG" ;;
    c  | action_col_name )                              needs_arg; action_col_name="$OPTARG" ;;
    p  | policy_num_col_name )                          needs_arg; policy_num_col_name="$OPTARG" ;;
    C  | calendar_t_col_name )                          needs_arg; calendar_t_col_name="$OPTARG" ;;
    U  | user_id_col_name )                             needs_arg; user_id_col_name="$OPTARG" ;;
    E  | action_prob_col_name )                         needs_arg; action_prob_col_name="$OPTARG" ;;
    P  | action_prob_func_filename )                    needs_arg; action_prob_func_filename="$OPTARG" ;;
    b  | action_prob_func_args_beta_index )             needs_arg; action_prob_func_args_beta_index="$OPTARG" ;;
    l  | rl_update_func_filename )                      needs_arg; rl_update_func_filename="$OPTARG" ;;
    Z  | rl_update_func_type )                          needs_arg; rl_update_func_type="$OPTARG" ;;
    B  | rl_update_func_args_beta_index )               needs_arg; rl_update_func_args_beta_index="$OPTARG" ;;
    D  | rl_update_func_args_action_prob_index )        needs_arg; rl_update_func_args_action_prob_index="$OPTARG" ;;
    j  | rl_update_func_args_action_prob_times_index )  needs_arg; rl_update_func_args_action_prob_times_index="$OPTARG" ;;
    I  | inference_loss_func_filename )                 needs_arg; inference_loss_func_filename="$OPTARG" ;;
    h  | inference_loss_func_args_theta_index )         needs_arg; inference_loss_func_args_theta_index="$OPTARG" ;;
    H  | theta_calculation_func_filename )              needs_arg; theta_calculation_func_filename="$OPTARG" ;;
    s  | seed )                                         needs_arg; seed="$OPTARG" ;;
    o  | only_analysis )                                needs_arg; only_analysis="$OPTARG" ;;
    Q  | suppress_interactive_data_checks )             needs_arg; suppress_interactive_data_checks="$OPTARG" ;;
    q  | suppress_all_data_checks )                     needs_arg; suppress_all_data_checks="$OPTARG" ;;
    z  | small_sample_correction )                      needs_arg; small_sample_correction="$OPTARG" ;;
    \? )                                        exit 2 ;;  # bad short option (error reported via getopts)
    * )                                         die "Illegal option --$OPT" ;; # bad long option
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list


save_dir_prefix="/n/netscratch/murphy_lab/Lab/nclosser/adaptive_sandwich_simulation_results/${SLURM_ARRAY_JOB_ID}"

if test -d save_dir_prefix; then
  die 'Output directory already exists. Please supply a unique label, perhaps a datetime.'
fi
save_dir="${save_dir_prefix}/${SLURM_ARRAY_TASK_ID}"
save_dir_glob="${save_dir_prefix}/*"
mkdir -p "$save_dir"

# Simulate an miwaves RL study (unless we just want to analyze previous results)
if [ "$only_analysis" -eq "0" ]; then
  echo "$(date +"%Y-%m-%d %T") miwaves_run_and_analysis_parallel: Beginning RL study simulation."
  python miwaves_sample_data/src/run_simulation.py \
    -p mixed_effects \
    -s $SLURM_ARRAY_TASK_ID \
    -d $save_dir
  echo "$(date +"%Y-%m-%d %T") miwaves_run_and_analysis_parallel: Finished RL study simulation."
fi

# Create a convenience variable that holds the output folder for the last script.
# This should really be output by that script or passed into it as an arg, but alas.
output_folder="miwaves_sample_data/results/num_users100_num_time_steps10_seed${seed}_delta_seed0_beta_mean[1]_beta_std[[1]]_gamma_std[[0.1]]_sigma_e20.1_policy_typemixed_effects"

# Create a convenience variable that holds the output folder for the last script
save_dir_suffix="simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering_RL}"
output_folder="${save_dir}/${save_dir_suffix}"
output_folder_glob="${save_dir_glob}/${save_dir_suffix}"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") miwaves_run_and_analysis_parallel: Beginning after-study analysis."
python after_study_analysis.py analyze-dataset \
  --study_df_pickle="${output_folder}/study_df.pkl" \
  --action_prob_func_filename=$action_prob_func_filename \
  --action_prob_func_args_pickle="${output_folder}/action_selection_function_dict.pkl" \
  --action_prob_func_args_beta_index=$action_prob_func_args_beta_index \
  --rl_update_func_filename=$rl_update_func_filename \
  --rl_update_func_type=$rl_update_func_type \
  --rl_update_func_args_pickle="${output_folder}/estimating_equation_function_dict.pkl" \
  --rl_update_func_args_beta_index=$rl_update_func_args_beta_index \
  --rl_update_func_args_action_prob_index=$rl_update_func_args_action_prob_index \
  --rl_update_func_args_action_prob_times_index=$rl_update_func_args_action_prob_times_index \
  --inference_loss_func_filename=$inference_loss_func_filename \
  --inference_loss_func_args_theta_index=$inference_loss_func_args_theta_index \
  --theta_calculation_func_filename=$theta_calculation_func_filename \
  --in_study_col_name=$in_study_col_name \
  --action_col_name=$action_col_name \
  --policy_num_col_name=$policy_num_col_name \
  --calendar_t_col_name=$calendar_t_col_name \
  --user_id_col_name=$user_id_col_name \
  --action_prob_col_name=$action_prob_col_name \
  --suppress_interactive_data_checks=$suppress_interactive_data_checks \
  --suppress_all_data_checks=$suppress_all_data_checks \
  --small_sample_correction=$small_sample_correction
echo $(date +"%Y-%m-%d %T") miwaves_run_and_analysis_parallel.sh: Finished after-study analysis.

echo $(date +"%Y-%m-%d %T") miwaves_run_and_analysis_parallel.sh: Simulation complete.
echo "$(date +"%Y-%m-%d %T") miwaves_run_and_analysis_parallel.sh: When all jobs have completed, you may collect and summarize the analyses with: bash simulation_collect_analyses.sh --input_glob=${output_folder_glob}/exp=1/analysis.pkl"
