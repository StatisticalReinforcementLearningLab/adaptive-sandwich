#!/bin/bash
set -eu

echo "$(date +"%Y-%m-%d %T") run_local_synthetic_thompson_sampling.sh: Beginning simulation."

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }


# Arguments that affect RL study simulation side
T=50
decisions_between_updates=1
update_cadence_offset=0
min_update_time=0
recruit_t=1 # How many UPDATES between recruitments
n=30
# recruit_n=$n is done below unless the user specifies recruit_n
synthetic_mode='delayed_1_action_dosage'
# synthetic_mode='delayed_1_dosage_paper'
# synthetic_mode='delayed_2_action_dosage'
# synthetic_mode='delayed_2_dosage_paper'
# synthetic_mode='delayed_5_action_dosage'
# synthetic_mode='delayed_5_dosage_paper'
steepness=5.0
RL_alg="smooth_posterior_sampling"
# RL_alg2="smooth_posterior_sampling"
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
action_centering_RL=0
lclip=0.1
uclip=0.9
dynamic_seeds=0
env_seed_override=-1
alg_seed_override=-1
# prior_mean="-0.37783337,0.18696958,2.3131008,0.32913807"
# prior_var_upper_triangle="1000000,0,0,0,1000000,0,0,1000000,0,1000000"
prior_mean="naive"
prior_var_upper_triangle="naive"
noise_var=1.0

# Arguments that only affect inference side.
in_study_col_name="in_study"
action_col_name="action"
policy_num_col_name="policy_num"
calendar_t_col_name="calendar_t"
user_id_col_name="user_id"
action_prob_col_name="action1prob"
action_prob_func_filename="functions_to_pass_to_analysis/smooth_thompson_sampling_act_prob_function_no_action_centering_partial.py" 
action_prob_func_args_beta_index=0
alg_update_func_filename="functions_to_pass_to_analysis/synthetic_BLR_estimating_function_no_action_centering_partial.py" # changed to a conditional function used in inference
alg_update_func_type="estimating" 
alg_update_func_args_beta_index=0
alg_update_func_args_action_prob_index=-1
alg_update_func_args_action_prob_times_index=-1
# inference_func_filename="functions_to_pass_to_analysis/synthetic_get_least_squares_loss_inference_no_action_centering.py"
inference_func_filename="functions_to_pass_to_analysis/inference_partial_linear_regression_0.py"
inference_func_args_theta_index=0
inference_func_type="loss"
# theta_calculation_func_filename="functions_to_pass_to_analysis/synthetic_estimate_theta_least_squares_no_action_centering.py"
theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_avg_reward_diff_partial_0.py"
suppress_interactive_data_checks=0
suppress_all_data_checks=0
small_sample_correction="none"
trim_small_singular_values=0

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts T:t:n:u:d:o:r:e:f:a:s:y:Y:A:G:i:c:p:C:U:P:b:l:Z:B:D:j:E:I:h:g:H:F:L:M:Q:q:z:J:K:O:w:-: OPT; do
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
    w  | trim_small_singular_values )                   needs_arg; trim_small_singular_values="$OPTARG" ;;

    \? )                                        exit 2 ;;  # bad short option (error reported via getopts)
    * )                                         die "Illegal option --$OPT" ;; # bad long option
  esac
done

shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Check for invalid options that do not start with a dash after parsing options.
for arg in "$@"; do
  if [[ "$arg" != -* ]]; then
    die "Invalid argument: $arg. Options must start with a dash (- or --)."
  fi
done

if [ -z "${recruit_n:-}" ]; then
  recruit_n=$n
fi


filename="_0"

# Simulate an RL study with the supplied arguments.  (We do just one repetition)
echo "$(date +"%Y-%m-%d %T") run_local_synthetic_thompson_sampling.sh: Beginning RL study simulation."
python rl_study_simulation_partial.py \
  --T=$T \
  --N=1 \
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
  --dynamic_seeds=$dynamic_seeds \
  --env_seed_override=$env_seed_override \
  --alg_seed_override=$alg_seed_override \
  --min_update_time=$min_update_time \
  --upper_clip=$uclip \
  --lower_clip=$lclip \
  --prior_mean=$prior_mean \
  --prior_var_upper_triangle=$prior_var_upper_triangle \
  --noise_var=$noise_var \
  --save_dir="n${n}_T${T}/0" \
  --Twoarmed=1 \
  --filename=$filename


# python rl_study_simulation_partial.py \
#   --T=$T \
#   --N=1 \
#   --n=$n \
#   --decisions_between_updates=$decisions_between_updates \
#   --update_cadence_offset=$update_cadence_offset \
#   --recruit_n=$recruit_n \
#   --recruit_t=$recruit_t \
#   --synthetic_mode=$synthetic_mode \
#   --steepness=$steepness \
#   --RL_alg=$RL_alg2 \
#   --err_corr=$err_corr \
#   --alg_state_feats=$alg_state_feats \
#   --action_centering=$action_centering_RL \
#   --dynamic_seeds=$dynamic_seeds \
#   --env_seed_override=$env_seed_override \
#   --alg_seed_override=$alg_seed_override \
#   --min_update_time=$min_update_time \
#   --upper_clip=$uclip \
#   --lower_clip=$lclip \
#   --prior_mean=$prior_mean \
#   --prior_var_upper_triangle=$prior_var_upper_triangle \
#   --noise_var=$noise_var \
#   --save_dir="n${n}_T${T}/0" \
#   --Z_id=0

echo "$(date +"%Y-%m-%d %T") run_local_synthetic_thompson_sampling.sh: Finished RL study simulation."

# Create a convenience variable that holds the output folder for the last script.
# This should really be output by that script or passed into it as an arg, but alas.
output_folder="n${n}_T${T}/0/simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_partial${filename}" # we set 0 because we focus on the first repetition
# output_folder_random="n${n}_T${T}/0/simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg2}_T=${T}_n=${n}_partial"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run_local_synthetic_thompson_sampling.sh: Beginning after-study analysis."
python after_study_analysis_partial.py analyze-dataset \
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
  --suppress_interactive_data_checks=$suppress_interactive_data_checks \
  --suppress_all_data_checks=$suppress_all_data_checks \
  --small_sample_correction=$small_sample_correction \
  --trim_small_singular_values=$trim_small_singular_values 

echo "$(date +"%Y-%m-%d %T") run_local_synthetic_thompson_sampling.sh: Ending after-study analysis."

echo "$(date +"%Y-%m-%d %T") run_local_synthetic_thompson_sampling.sh: Finished simulation."