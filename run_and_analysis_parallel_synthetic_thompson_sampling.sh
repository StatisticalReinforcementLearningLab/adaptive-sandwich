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
# sbatch --array=[0-99] run_and_analysis_parallel_synthetic_thompson_sampling.sh --T=25 --n=100 --recruit_n=100 --recruit_t=1

# To analyze, run simulation_collect_analyses.sh as described in the
# output of one of the simulation runs.

# If running on GPU, the following can be used:
# S BATCH -p gpu_requeue                                                       # Target Partition
# S BATCH --gres=gpu:1                                                         # Request a GPU

# Stop on nonzero exit codes and use of undefined variables
set -eu

echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Parsing options.

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# Arguments that affect RL study simulation side
T=140
decisions_between_updates=14
update_cadence_offset=0
min_update_time=14
recruit_t=2 # How many UPDATES between recruitments
n=100
# recruit_n=$n is done below unless the user specifies recruit_n
# synthetic_mode='delayed_1_action_dosage'
# synthetic_mode='delayed_1_dosage_paper'
# synthetic_mode='delayed_2_action_dosage'
# synthetic_mode='delayed_2_dosage_paper'
synthetic_mode='delayed_5_action_dosage'
# synthetic_mode='delayed_5_dosage_paper'
steepness=1.5
RL_alg="smooth_posterior_sampling"
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
action_centering_RL=0
lclip=0.1
uclip=0.9
dynamic_seeds=0
env_seed_override=-1
alg_seed_override=-1
# prior_mean="-0.37783337,0.18696958,2.3131008,0.32913807"
prior_mean="naive"
prior_var_upper_triangle="naive"
noise_var=1.0

# Arguments that only affect inference side.
# Arguments that only affect inference side.
in_study_col_name="in_study"
action_col_name="action"
policy_num_col_name="policy_num"
calendar_t_col_name="calendar_t"
user_id_col_name="user_id"
action_prob_col_name="action1prob"
reward_col_name="reward"
action_prob_func_filename="functions_to_pass_to_analysis/smooth_thompson_sampling_act_prob_function_no_action_centering.py"
action_prob_func_args_beta_index=0
alg_update_func_filename="functions_to_pass_to_analysis/synthetic_BLR_estimating_function_no_action_centering.py"
alg_update_func_type="estimating"
alg_update_func_args_beta_index=0
alg_update_func_args_action_prob_index=-1
alg_update_func_args_action_prob_times_index=-1
inference_func_filename="functions_to_pass_to_analysis/synthetic_get_least_squares_loss_inference_no_action_centering.py"
inference_func_args_theta_index=0
inference_func_type="loss"
theta_calculation_func_filename="functions_to_pass_to_analysis/synthetic_estimate_theta_least_squares_no_action_centering.py"
suppress_interactive_data_checks=1
suppress_all_data_checks=0
small_sample_correction="none"
adaptive_bread_inverse_stabilization_method="add_ridge_fixed_condition_number"

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
# Note that the N argument is not supplied here: the number of simulations is
# determined by the number of jobs in the slurm job array.
while getopts T:t:n:u:d:o:r:e:f:a:s:y:Y:A:G:i:c:p:C:U:E:X:P:b:l:Z:B:D:j:I:h:g:H:F:L:M:Q:q:z:J:K:O:w:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    T  | max_time )                                     needs_arg; T="$OPTARG" ;;
    t  | recruit_t )                                    needs_arg; recruit_t="$OPTARG" ;;
    n  | num_users )                                    needs_arg; n="$OPTARG" ;;
    u  | recruit_n )                                    needs_arg; recruit_n="$OPTARG" ;;
    d  | decisions_between_updates )                    needs_arg; decisions_between_updates="$OPTARG" ;;
    o  | update_cadence_offset )                        needs_arg; update_cadence_offset="$OPTARG" ;;
    r  | RL_alg )                                       needs_arg; RL_alg="$OPTARG" ;;
    e  | err_corr )                                     needs_arg; err_corr="$OPTARG" ;;
    f  | alg_state_feats )                              needs_arg; alg_state_feats="$OPTARG" ;;
    a  | action_centering_RL )                          needs_arg; action_centering_RL="$OPTARG" ;;
    s  | steepness )                                    needs_arg; steepness="$OPTARG" ;;
    y  | synthetic_mode )                               needs_arg; synthetic_mode="$OPTARG" ;;
    Y  | min_update_time )                              needs_arg; min_update_time="$OPTARG" ;;
    A  | uclip )                                        needs_arg; uclip="$OPTARG" ;;
    G  | lclip )                                        needs_arg; lclip="$OPTARG" ;;
    i  | in_study_col_name )                            needs_arg; in_study_col_name="$OPTARG" ;;
    c  | action_col_name )                              needs_arg; action_col_name="$OPTARG" ;;
    p  | policy_num_col_name )                          needs_arg; policy_num_col_name="$OPTARG" ;;
    C  | calendar_t_col_name )                          needs_arg; calendar_t_col_name="$OPTARG" ;;
    U  | user_id_col_name )                             needs_arg; user_id_col_name="$OPTARG" ;;
    E  | action_prob_col_name )                         needs_arg; action_prob_col_name="$OPTARG" ;;
    X  | reward_col_name )                              needs_arg; reward_col_name="$OPTARG" ;;
    P  | action_prob_func_filename )                    needs_arg; action_prob_func_filename="$OPTARG" ;;
    b  | action_prob_func_args_beta_index )             needs_arg; action_prob_func_args_beta_index="$OPTARG" ;;
    l  | alg_update_func_filename )                     needs_arg; alg_update_func_filename="$OPTARG" ;;
    Z  | alg_update_func_type )                         needs_arg; alg_update_func_type="$OPTARG" ;;
    B  | alg_update_func_args_beta_index )              needs_arg; alg_update_func_args_beta_index="$OPTARG" ;;
    D  | alg_update_func_args_action_prob_index )       needs_arg; alg_update_func_args_action_prob_index="$OPTARG" ;;
    j  | alg_update_func_args_action_prob_times_index ) needs_arg; alg_update_func_args_action_prob_times_index="$OPTARG" ;;
    I  | inference_func_filename )                      needs_arg; inference_func_filename="$OPTARG" ;;
    h  | inference_func_args_theta_index )              needs_arg; inference_func_args_theta_index="$OPTARG" ;;
    g  | inference_func_type )                          needs_arg; inference_func_type="$OPTARG" ;;
    H  | theta_calculation_func_filename )              needs_arg; theta_calculation_func_filename="$OPTARG" ;;
    F  | dynamic_seeds )                                needs_arg; dynamic_seeds="$OPTARG" ;;
    L  | env_seed_override )                            needs_arg; env_seed_override="$OPTARG" ;;
    M  | alg_seed_override )                            needs_arg; alg_seed_override="$OPTARG" ;;
    Q  | suppress_interactive_data_checks )             needs_arg; suppress_interactive_data_checks="$OPTARG" ;;
    q  | suppress_all_data_checks )                     needs_arg; suppress_all_data_checks="$OPTARG" ;;
    z  | small_sample_correction )                      needs_arg; small_sample_correction="$OPTARG" ;;
    J  | prior_mean )                                   needs_arg; prior_mean="$OPTARG" ;;
    K  | prior_var_upper_triangle )                     needs_arg; prior_var_upper_triangle="$OPTARG" ;;
    O  | noise_var )                                    needs_arg; noise_var="$OPTARG" ;;
    w  | adaptive_bread_inverse_stabilization_method )  needs_arg; adaptive_bread_inverse_stabilization_method="$OPTARG" ;;
    \? )                                        exit 2 ;;  # bad short option (error reported via getopts)
    * )                                         die "Illegal long option --$OPT" ;; # bad long option
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

if [ -z "${recruit_n:-}" ]; then
  recruit_n=$n
fi

# Load Python 3.10, among other things
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Loading mamba and CUDA modules.
module load Mambaforge/22.11.1-fasrc01
# if using GPU, something like the following will be necessary:
# module load cuda/12.2.0-fasrc01

# Make virtualenv if necessary, and then activate it
cd ~
if ! test -d venv; then
  echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Creating venv, as it did not exist.
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd ~/adaptive-sandwich
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Making sure Python requirements are installed.
pip install -r requirements.txt
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: All Python requirements installed.

save_dir_prefix="/n/netscratch/murphy_lab/Lab/nclosser/adaptive_sandwich_simulation_results/${SLURM_ARRAY_JOB_ID}"

if test -d save_dir_prefix; then
  die 'Output directory already exists. Please supply a unique label, perhaps a datetime.'
fi
save_dir="${save_dir_prefix}/${SLURM_ARRAY_TASK_ID}"
save_dir_glob="${save_dir_prefix}/*"
mkdir -p "$save_dir"

# Simulate an RL study with the supplied arguments.  (We do just one repetition)
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Beginning RL simulations.
python rl_study_simulation.py \
  --T=$T \
  --N=1 \
  --parallel_task_index=$SLURM_ARRAY_TASK_ID \
  --n=$n \
  --decisions_between_updates=$decisions_between_updates \
  --update_cadence_offset=$update_cadence_offset \
  --recruit_n=$recruit_n \
  --recruit_t=$recruit_t \
  --synthetic_mode=$synthetic_mode \
  --steepness=$steepness \
  --RL_alg=$RL_alg \
  --err_corr=$err_corr \
  --alg_state_feats=$alg_state_feats \
  --action_centering=$action_centering_RL \
  --save_dir=$save_dir \
  --dynamic_seeds=$dynamic_seeds \
  --env_seed_override=$env_seed_override \
  --alg_seed_override=$alg_seed_override \
  --min_update_time=$min_update_time \
  --upper_clip=$uclip \
  --lower_clip=$lclip \
  --prior_mean=$prior_mean \
  --prior_var_upper_triangle=$prior_var_upper_triangle \
  --noise_var=$noise_var
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Finished RL simulations.

# Create a convenience variable that holds the output folder for the last script
save_dir_suffix="simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering_RL}"
output_folder="${save_dir}/${save_dir_suffix}"
output_folder_glob="${save_dir_glob}/${save_dir_suffix}"

# Analyze dataset created in the above simulation
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Beginning after-study analysis.
python after_study_analysis.py analyze-dataset \
  --study_df_pickle="${output_folder}/exp=1/study_df.pkl" \
  --action_prob_func_filename=$action_prob_func_filename \
  --action_prob_func_args_pickle="${output_folder}/exp=1/pi_args.pkl" \
  --action_prob_func_args_beta_index=$action_prob_func_args_beta_index \
  --alg_update_func_filename=$alg_update_func_filename \
  --alg_update_func_type=$alg_update_func_type \
  --alg_update_func_args_pickle="${output_folder}/exp=1/rl_update_args.pkl" \
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
  --reward_col_name=$reward_col_name \
  --suppress_interactive_data_checks=$suppress_interactive_data_checks \
  --suppress_all_data_checks=$suppress_all_data_checks \
  --small_sample_correction=$small_sample_correction \
  --adaptive_bread_inverse_stabilization_method=$adaptive_bread_inverse_stabilization_method
echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Finished after-study analysis.

echo $(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: Simulation complete.
echo "$(date +"%Y-%m-%d %T") run_and_analysis_parallel_synthetic_thompson_sampling.sh: When all jobs have completed, you may collect and summarize the analyses with: bash simulation_collect_analyses.sh --input_glob=${output_folder_glob}/exp=1/analysis.pkl --num_users=$n [--index_to_check_ci_coverage=<>]  --in_study_col_name=$in_study_col_name --action_col_name=$action_col_name --action_prob_col_name=$action_prob_col_name"
