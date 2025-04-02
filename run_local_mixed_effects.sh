#!/bin/bash
set -eu

echo "$(date +"%Y-%m-%d %T") run_local_mixed_effects.sh: Beginning simulation."

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# Arguments that only affect simulation side.
num_users=100
num_time_steps=10
seed=0
delta_seed=0
beta_mean=1
beta_var=1
gamma_var=.1
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
action_prob_func_filename="functions_to_pass_to_analysis/miwaves_action_selection.py"
action_prob_func_args_beta_index=0
alg_update_func_filename="functions_to_pass_to_analysis/miwaves_RL_estimating_function.py"
alg_update_func_type="estimating"
alg_update_func_args_beta_index=0
alg_update_func_args_action_prob_index=-1
alg_update_func_args_action_prob_times_index=-1
inference_func_filename="functions_to_pass_to_analysis/miwaves_primary_analysis_loss.py"
inference_func_args_theta_index=0
inference_func_type="loss"
theta_calculation_func_filename="functions_to_pass_to_analysis/miwaves_estimate_theta_primary_analysis.py"
suppress_interactive_data_checks=0
suppress_all_data_checks=0
small_sample_correction="none"

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts m:T:s:S:G:t:g:e:O:o:i:c:p:C:U:E:P:b:l:Z:B:D:j:I:h:g:H:Q:q:z:-: OPT; do
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
    t  | beta_var )                                     needs_arg; beta_var="$OPTARG" ;;
    g  | gamma_var )                                    needs_arg; gamma_var="$OPTARG" ;;
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
    g  | inference_func_type )                          needs_arg; inference_func_type="$OPTARG" ;;
    H  | theta_calculation_func_filename )              needs_arg; theta_calculation_func_filename="$OPTARG" ;;
    Q  | suppress_interactive_data_checks )             needs_arg; suppress_interactive_data_checks="$OPTARG" ;;
    q  | suppress_all_data_checks )                     needs_arg; suppress_all_data_checks="$OPTARG" ;;
    z  | small_sample_correction )                      needs_arg; small_sample_correction="$OPTARG" ;;
    \? )                                        exit 2 ;;  # bad short option (error reported via getopts)
    * )                                         die "Illegal option --$OPT" ;; # bad long option
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Simulate an miwaves RL study (unless we just want to analyze previous results)
if [ "$only_analysis" -eq "0" ]; then
  echo "$(date +"%Y-%m-%d %T") run_local_mixed_effects.sh: Beginning RL study simulation."
  python mixed_effects_sample_data/src/run_simulation.py \
    --num_users $num_users \
    --num_time_steps $num_time_steps \
    --seed $seed \
    --delta_seed $delta_seed \
    --beta_mean $beta_mean \
    --beta_var $beta_var \
    --gamma_var $gamma_var \
    --sigma_e2 $sigma_e2 \
    --policy_type $policy_type
  echo "$(date +"%Y-%m-%d %T") run_local_mixed_effects.sh: Finished RL study simulation."
fi

# Create a convenience variable that holds the output folder for the last script.
# This should really be output by that script or passed into it as an arg, but alas.
output_folder="mixed_effects_sample_data/results/num_users${num_users}_num_time_steps${num_time_steps}_seed${seed}_delta_seed0_beta_mean[$(printf '%.1f' $beta_mean)]_beta_var[[$(printf '%.1f' $beta_var)]]_gamma_var[[$(printf '%.1f' $gamma_var)]]_sigma_e2$(printf '%.1f' $sigma_e2)_policy_typemixed_effects"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run_local_mixed_effects.sh: Beginning after-study analysis."
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
  --small_sample_correction=$small_sample_correction
echo "$(date +"%Y-%m-%d %T") run_local_mixed_effects.sh: Ending after-study analysis."

echo "$(date +"%Y-%m-%d %T") run_local_mixed_effects.sh: Finished simulation."
