#!/bin/bash
set -eu

echo "$(date +"%Y-%m-%d %T") run.sh: Beginning simulation."

T=25
decisions_between_updates=2
# recruit_n=25; recruit_t=2
recruit_n=50; recruit_t=1
n=50
min_users=1
synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_01_5_dosage'
#synthetic_mode='test_1_1_T2'
#synthetic_mode='delayed_effects_large'
steepness=0.0
eta=0
RL_alg="sigmoid_LS"
#RL_alg="posterior_sampling"
#RL_alg="fixed_randomization"
#err_corr='independent'
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
action_centering=0
#TODO: not used currently but maybe should be
debug=0
redo_analyses=1

# Simulate an RL study with the supplied arguments.  (We do just one repetition)
echo "$(date +"%Y-%m-%d %T") run.sh: Beginning RL study simulation."
python rl_study_simulation.py --T=$T --N=1 --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering --profile
echo "$(date +"%Y-%m-%d %T") run.sh: Finished RL study simulation."

# Create a convenience variable that holds the output folder for the last script
output_folder="simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering}"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run.sh: Beginning after-study analysis."
python after_study_analysis.py analyze-dataset --study_dataframe_pickle="${output_folder}/exp=1/study_df.pkl" --rl_algorithm_object_pickle="${output_folder}/exp=1/study_RLalg.pkl" --profile
echo "$(date +"%Y-%m-%d %T") run.sh: Ending after-study analysis."

echo "$(date +"%Y-%m-%d %T") run.sh: Finished simulation."
