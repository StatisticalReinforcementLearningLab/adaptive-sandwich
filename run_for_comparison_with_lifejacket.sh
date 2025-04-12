set -eu

# We run two experiments because there's a bug so that N=1 blows up.
# But we only need to use one of the results. The suggested analysis using the
# package below looks at exp=1.
N=2

n=50
#n=100
#n=500

#synthetic_mode='delayed_1_action_dosage'
# synthetic_mode='delayed_1_dosage_paper'
synthetic_mode='delayed_5_action_dosage'
# synthetic_mode='delayed_5_dosage_paper'

# steepness=0.5
# steepness=1
steepness=5

recruit_n=$n
T=2
decisions_between_updates=1
min_users=1
RL_alg="sigmoid_LS"
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
inference_mode="model"
recruit_t=1
debug=0

# Don't do anything
eta=0
action_centering=0

# Run experiment, saving output suiting for both Kelly's analysis and Nowell's.
python RL_Study_Simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering

# Run Kelly's analysis.
python After_Study_Analyses.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --debug $debug --RL_alg $RL_alg --eta $eta --alg_state_feats $alg_state_feats --inference_mode $inference_mode --action_centering $action_centering

echo "Kelly's analysis complete.  The following command will run Nowell's analysis on a suitable branch (probably main):"

output_folder="simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=0"
in_study_col_name="in_study"
action_col_name="action"
policy_num_col_name="policy_num"
calendar_t_col_name="calendar_t"
user_id_col_name="user_id"
action_prob_col_name="action1prob"
action_prob_func_filename="functions_to_pass_to_analysis/synthetic_get_action_1_prob_pure.py"
action_prob_func_args_beta_index=0
alg_update_func_filename="functions_to_pass_to_analysis/synthetic_get_least_squares_loss_rl.py"
alg_update_func_type="loss"
alg_update_func_args_beta_index=0
alg_update_func_args_action_prob_index=5
alg_update_func_args_action_prob_times_index=6
inference_func_filename="functions_to_pass_to_analysis/synthetic_get_least_squares_loss_inference_no_action_centering.py"
inference_func_args_theta_index=0
inference_func_type="loss"
theta_calculation_func_filename="functions_to_pass_to_analysis/synthetic_estimate_theta_least_squares_no_action_centering.py"
suppress_interactive_data_checks=0
suppress_all_data_checks=0
small_sample_correction="none"

echo python after_study_analysis.py analyze-dataset \
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
  --small_sample_correction=$small_sample_correction
