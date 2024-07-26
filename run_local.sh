#!/bin/bash
set -eu

echo "$(date +"%Y-%m-%d %T") run_local.sh: Beginning simulation."

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }


# Arguments that affect RL study simulation side
T=10
decisions_between_updates=1
recruit_t=1
n=100
recruit_n=$n
min_users=1
synthetic_mode='delayed_1_dosage'
steepness=0.0
RL_alg="sigmoid_LS"
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
action_centering_RL=0

# Arguments that only affect inference side.
in_study_col_name="in_study"
action_col_name="action"
policy_num_col_name="policy_num"
calendar_t_col_name="calendar_t"
user_id_col_name="user_id"
action_prob_col_name="action1prob"
action_prob_func_filename="functions_to_pass_to_analysis/get_action_1_prob_pure.py"
action_prob_func_args_beta_index=0
rl_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_rl.py"
rl_loss_func_args_beta_index=0
rl_loss_func_args_action_prob_index=5
inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py"
inference_loss_func_args_theta_index=0
theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts T:t:n:u:d:m:r:e:f:a:s:y:i:c:p:C:U:P:b:l:B:D:E:I:h:H:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    T  | max_time )                             needs_arg; T="$OPTARG" ;;
    t  | recruit_t )                            needs_arg; recruit_t="$OPTARG" ;;
    n  | num_users )                            needs_arg; n="$OPTARG" ;;
    u  | recruit_n )                            needs_arg; recruit_n="$OPTARG" ;;
    d  | decisions_between_updates )            needs_arg; decisions_between_updates="$OPTARG" ;;
    m  | min_users )                            needs_arg; min_users="$OPTARG" ;;
    r  | RL_alg )                               needs_arg; RL_alg="$OPTARG" ;;
    e  | err_corr )                             needs_arg; err_corr="$OPTARG" ;;
    f  | alg_state_feats )                      needs_arg; alg_state_feats="$OPTARG" ;;
    a  | action_centering_RL )                  needs_arg; action_centering_RL="$OPTARG" ;;
    s  | steepness )                            needs_arg; steepness="$OPTARG" ;;
    y  | synthetic_mode )                       needs_arg; synthetic_mode="$OPTARG" ;;
    i  | in_study_col_name )                    needs_arg; in_study_col_name="$OPTARG" ;;
    c  | action_col_name )                      needs_arg; action_col_name="$OPTARG" ;;
    p  | policy_num_col_name )                  needs_arg; policy_num_col_name="$OPTARG" ;;
    C  | calendar_t_col_name )                  needs_arg; calendar_t_col_name="$OPTARG" ;;
    U  | user_id_col_name )                     needs_arg; user_id_col_name="$OPTARG" ;;
    E  | action_prob_col_name )                 needs_arg; action_prob_col_name="$OPTARG" ;;
    P  | action_prob_func_filename )            needs_arg; action_prob_func_filename="$OPTARG" ;;
    b  | action_prob_func_args_beta_index )     needs_arg; action_prob_func_args_beta_index="$OPTARG" ;;
    l  | rl_loss_func_filename )                needs_arg; rl_loss_func_filename="$OPTARG" ;;
    B  | rl_loss_func_args_beta_index )         needs_arg; rl_loss_func_args_beta_index="$OPTARG" ;;
    D  | rl_loss_func_args_action_prob_index )  needs_arg; rl_loss_func_args_action_prob_index="$OPTARG" ;;
    I  | inference_loss_func_filename )         needs_arg; inference_loss_func_filename="$OPTARG" ;;
    h  | inference_loss_func_args_theta_index ) needs_arg; inference_loss_func_args_theta_index="$OPTARG" ;;
    H  | theta_calculation_func_filename )      needs_arg; theta_calculation_func_filename="$OPTARG" ;;
    \? )                                        exit 2 ;;  # bad short option (error reported via getopts)
    * )                                         die "Illegal option --$OPT" ;; # bad long option
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Simulate an RL study with the supplied arguments.  (We do just one repetition)
echo "$(date +"%Y-%m-%d %T") run_local.sh: Beginning RL study simulation."
python rl_study_simulation.py \
  --T=$T \
  --N=1 \
  --n=$n \
  --min_users=$min_users \
  --decisions_between_updates $decisions_between_updates \
  --recruit_n $recruit_n \
  --recruit_t $recruit_t \
  --synthetic_mode $synthetic_mode \
  --steepness $steepness \
  --RL_alg $RL_alg \
  --err_corr $err_corr \
  --alg_state_feats $alg_state_feats \
  --action_centering $action_centering_RL
echo "$(date +"%Y-%m-%d %T") run_local.sh: Finished RL study simulation."

# Create a convenience variable that holds the output folder for the last script.
# This should really be output by that script or passed into it as an arg, but alas.
output_folder="simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering_RL}"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run_local.sh: Beginning after-study analysis."
python after_study_analysis.py analyze-dataset \
  --study_df_pickle="${output_folder}/exp=1/study_df.pkl" \
  --action_prob_func_filename=$action_prob_func_filename \
  --action_prob_func_args_pickle="${output_folder}/exp=1/pi_args.pkl" \
  --action_prob_func_args_beta_index=$action_prob_func_args_beta_index \
  --rl_loss_func_filename=$rl_loss_func_filename \
  --rl_loss_func_args_pickle="${output_folder}/exp=1/rl_update_args.pkl" \
  --rl_loss_func_args_beta_index=$rl_loss_func_args_beta_index \
  --rl_loss_func_args_action_prob_index=$rl_loss_func_args_action_prob_index \
  --inference_loss_func_filename=$inference_loss_func_filename \
  --inference_loss_func_args_theta_index=$inference_loss_func_args_theta_index \
  --theta_calculation_func_filename=$theta_calculation_func_filename \
  --in_study_col_name=$in_study_col_name \
  --action_col_name=$action_col_name \
  --policy_num_col_name=$policy_num_col_name \
  --calendar_t_col_name=$calendar_t_col_name \
  --user_id_col_name=$user_id_col_name \
  --action_prob_col_name=$action_prob_col_name
echo "$(date +"%Y-%m-%d %T") run_local.sh: Ending after-study analysis."

echo "$(date +"%Y-%m-%d %T") run_local.sh: Finished simulation."
