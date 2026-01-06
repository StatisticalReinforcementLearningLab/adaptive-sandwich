#!/bin/bash
set -eu

echo "$(date +"%Y-%m-%d %T") run_local_Miwaves_thompson_sampling.sh: Beginning simulation."

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }


# Arguments that affect RL study simulation side
T=60 # 60
decisions_between_updates=1 ##### important
update_cadence_offset=0
min_update_time=0
recruit_t=1 # How many UPDATES between recruitments
n=30 # 

steepness=21.05 # 20/0.95
RL_alg="smooth_posterior_sampling"
alg_state_feats="intercept,S1,S2,S3" # design state
action_centering_RL=0
lclip=0.2
uclip=0.8
dynamic_seeds=0
env_seed_override=-1
alg_seed_override=-1
prior_mean="2.12,0.0,0.0,-0.69,0.0,0.0,0.0,0.0" # Dim=8 for no action-centering; Dim=12 for action-centering; double everything if considering the intersection terms between states
prior_var_upper_triangle="0.6084,0,0,0,0,0,0,0,0.1444,0,0,0,0,0,0,0.3844,0,0,0,0,0,0.9604,0,0,0,0,0.0729,0,0,0,0.1089,0,0,0.09,0,0.1024" # 8+7+6+5+4+3+2+1=36 elements for no action-centering
noise_var=1.0
act_cost_threshold=0.1 # need to be changed

# Arguments that only affect inference side.
in_study_col_name="in_study"
action_col_name="action"
policy_num_col_name="policy_num"
calendar_t_col_name="calendar_t"
user_id_col_name="user_id"
action_prob_col_name="action1prob"
reward_col_name="reward"
action_prob_func_filename="functions_to_pass_to_analysis/smooth_thompson_sampling_act_prob_function_no_action_centering_Miwaves.py"
action_prob_func_args_beta_index=0
alg_update_func_filename="functions_to_pass_to_analysis/Miwaves_BLR_estimating_function_no_action_centering_twoarmed.py"
alg_update_func_type="estimating"
alg_update_func_args_beta_index=0
alg_update_func_args_action_prob_index=-1
alg_update_func_args_action_prob_times_index=-1
alg_update_func_args_previous_betas_index=-1 # for recursive algorithms; -1 if not used
inference_func_filename="functions_to_pass_to_analysis/primary_analysis_avg_reward_sum_loss.py"
inference_func_args_theta_index=0
inference_func_type="loss"
theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_avg_reward_sum.py"
suppress_interactive_data_checks=0
suppress_all_data_checks=0
small_sample_correction="none"
trim_small_singular_values=0
collect_data_for_blowup_supervised_learning=0
stabilize_joint_adaptive_bread_inverse=0

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts T:t:n:u:d:o:r:e:f:a:s:y:Y:A:G:i:c:p:C:U:E:X:P:b:l:Z:B:D:j:I:h:g:H:F:L:M:Q:q:z:J:K:O:k:m:-: OPT; do
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
    f  | alg_state_feats )                              needs_arg; alg_state_feats="$OPTARG" ;;
    a  | action_centering_RL )                          needs_arg; action_centering_RL="$OPTARG" ;;
    s  | steepness )                                    needs_arg; steepness="$OPTARG" ;;
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
    R  | alg_update_func_args_previous_betas_index )    needs_arg; alg_update_func_args_previous_betas_index="$OPTARG" ;;
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
    k  | collect_data_for_blowup_supervised_learning )  needs_arg; collect_data_for_blowup_supervised_learning="$OPTARG" ;;
    m  | stabilize_joint_adaptive_bread_inverse )       needs_arg; stabilize_joint_adaptive_bread_inverse="$OPTARG" ;;

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

filename="_averagerewards"

# Simulate an RL study with the supplied arguments.  (We do just one repetition)
echo "$(date +"%Y-%m-%d %T") run_local_Miwaves_thompson_sampling.sh: Beginning RL study simulation."
# python rl_study_simulation.py \
python rl_study_simulation_partial_Miwaves.py \
  --T=$T \
  --N=1 \
  --n=$n \
  --dataset_type="miwaves" \
  --decisions_between_updates=$decisions_between_updates \
  --update_cadence_offset=$update_cadence_offset \
  --recruit_n=$recruit_n \
  --recruit_t=$recruit_t \
  --steepness=$steepness \
  --RL_alg=$RL_alg \
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
  --Twoarmed=0 \
  --filename=$filename \
  --act_cost_threshold=$act_cost_threshold

echo "$(date +"%Y-%m-%d %T") run_local_Miwaves_thompson_sampling.sh: Finished RL study simulation."

# Create a convenience variable that holds the output folder for the last script.
# This should really be output by that script or passed into it as an arg, but alas.
output_folder="n${n}_T${T}/0/simulated_data/miwaves_alg=${RL_alg}_T=${T}_n=${n}${filename}"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run_local_Miwaves_thompson_sampling.sh: Beginning after-study analysis."

######### Using the package
lifejacket analyze \
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
  --alg_update_func_args_previous_betas_index=$alg_update_func_args_previous_betas_index \
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
  --collect_data_for_blowup_supervised_learning=$collect_data_for_blowup_supervised_learning \
  --stabilize_joint_adaptive_bread_inverse=$stabilize_joint_adaptive_bread_inverse

######### Using the old implementation
# python after_study_analysis_partial.py analyze-dataset \
#   --study_df_pickle="${output_folder}/exp=1/study_df.pkl" \
#   --action_prob_func_filename=$action_prob_func_filename \
#   --action_prob_func_args_pickle="${output_folder}/exp=1/pi_args.pkl" \
#   --action_prob_func_args_beta_index=$action_prob_func_args_beta_index \
#   --alg_update_func_filename=$alg_update_func_filename \
#   --alg_update_func_type=$alg_update_func_type \
#   --alg_update_func_args_pickle="${output_folder}/exp=1/rl_update_args.pkl" \
#   --alg_update_func_args_beta_index=$alg_update_func_args_beta_index \
#   --alg_update_func_args_action_prob_index=$alg_update_func_args_action_prob_index \
#   --alg_update_func_args_action_prob_times_index=$alg_update_func_args_action_prob_times_index \
#   --inference_func_filename=$inference_func_filename \
#   --inference_func_args_theta_index=$inference_func_args_theta_index \
#   --inference_func_type=$inference_func_type \
#   --theta_calculation_func_filename=$theta_calculation_func_filename \
#   --in_study_col_name=$in_study_col_name \
#   --action_col_name=$action_col_name \
#   --policy_num_col_name=$policy_num_col_name \
#   --calendar_t_col_name=$calendar_t_col_name \
#   --user_id_col_name=$user_id_col_name \
#   --action_prob_col_name=$action_prob_col_name \
#   --suppress_interactive_data_checks=$suppress_interactive_data_checks \
#   --suppress_all_data_checks=$suppress_all_data_checks \
#   --small_sample_correction=$small_sample_correction \
#   --trim_small_singular_values=$trim_small_singular_values 

echo "$(date +"%Y-%m-%d %T") run_local_Miwaves_thompson_sampling.sh: Ending after-study analysis."

echo "$(date +"%Y-%m-%d %T") run_local_Miwaves_thompson_sampling.sh: Finished simulation."