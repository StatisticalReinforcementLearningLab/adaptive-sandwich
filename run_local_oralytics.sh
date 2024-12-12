#!/bin/bash
set -eu

echo "$(date +"%Y-%m-%d %T") run_local_oralytics.sh: Beginning simulation."

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# Arguments that only affect simulation side.
seed=0
only_analysis=0

# Arguments that only affect inference side.
in_study_col_name="in_study_indicator"
action_col_name="action"
policy_num_col_name="policy_idx"
calendar_t_col_name="calendar_decision_t"
user_id_col_name="user_idx"
action_prob_col_name="act_prob"
action_prob_func_filename="functions_to_pass_to_analysis/oralytics_act_prob_function.py"
action_prob_func_args_beta_index=0
rl_update_func_filename="functions_to_pass_to_analysis/oralytics_RL_estimating_function.py"
rl_update_func_type="estimating"
rl_update_func_args_beta_index=0
rl_update_func_args_action_prob_index=4
rl_update_func_args_action_prob_times_index=5
# inference_loss_func_filename="functions_to_pass_to_analysis/oralytics_primary_analysis_loss.py"
inference_loss_func_filename="functions_to_pass_to_analysis/oralytics_primary_analysis_avg_reward_sum_loss_debug.py"
inference_loss_func_args_theta_index=0
# theta_calculation_func_filename="functions_to_pass_to_analysis/oralytics_estimate_theta_primary_analysis.py"
theta_calculation_func_filename="functions_to_pass_to_analysis/oralytics_estimate_theta_primary_analysis_avg_reward_sum_debug.py"
suppress_interactive_data_checks=0
suppress_all_data_checks=0

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts i:c:p:C:U:E:P:b:l:Z:B:D:j:I:h:H:s:o:Q:q:-: OPT; do
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
    \? )                                        exit 2 ;;  # bad short option (error reported via getopts)
    * )                                         die "Illegal option --$OPT" ;; # bad long option
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Simulate an oralytics RL study (unless we just want to analyze previous results)
if [ "$only_analysis" -eq "0" ]; then
  echo "$(date +"%Y-%m-%d %T") run_local_oralytics.sh: Beginning RL study simulation."
  python oralytics_sample_data/Archive/src/run_exps.py ${seed}
  echo "$(date +"%Y-%m-%d %T") run_local_oralytics.sh: Finished RL study simulation."
fi

# Create a convenience variable that holds the output folder for the last script.
# This should really be output by that script or passed into it as an arg, but alas.
output_folder="oralytics_sample_data/Archive/exps/write/NON_STAT_LOW_R_None_0.515_14_full_pooling/${seed}"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run_local_oralytics.sh: Beginning after-study analysis."
python after_study_analysis.py analyze-dataset \
  --study_df_pickle="${output_folder}/${seed}_study_data.pkl" \
  --action_prob_func_filename=$action_prob_func_filename \
  --action_prob_func_args_pickle="${output_folder}/${seed}_action_data.pkl" \
  --action_prob_func_args_beta_index=$action_prob_func_args_beta_index \
  --rl_update_func_filename=$rl_update_func_filename \
  --rl_update_func_type=$rl_update_func_type \
  --rl_update_func_args_pickle="${output_folder}/${seed}_loss_fn_data.pkl" \
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
  --suppress_all_data_checks=$suppress_all_data_checks
echo "$(date +"%Y-%m-%d %T") run_local_oralytics.sh: Ending after-study analysis."

echo "$(date +"%Y-%m-%d %T") run_local_oralytics.sh: Finished simulation."
