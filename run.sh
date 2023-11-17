#!/bin/bash
set -e

# T=25,50
# Steepness=0.5,1,2?
# Sample size=100,500,1000

T=25
decisions_between_updates=2
# recruit_n=25; recruit_t=2
recruit_n=50; recruit_t=1
N=2
n=50
min_users=1
synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_01_5_dosage'
#synthetic_mode='test_1_1_T2'
#synthetic_mode='delayed_effects_large'
steepness=1
eta=0
RL_alg="sigmoid_LS"
#RL_alg="posterior_sampling"
#RL_alg="fixed_randomization"
#err_corr='independent'
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
inference_mode="model"
action_centering=0

debug=0
# TODO Consider this redo analyses argument to add
redo_analyses=1

# Need to dynamically construct path to second script based on args in first
# python RL_Study_Simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering
for i in $(seq 1 $N)
do
   echo $i
   python after_study_analysis.py --study_dataframe_pickle="simulated_data/synthetic_mode=delayed_1_dosage_alg=sigmoid_LS_T=25_n=50_recruitN=50_decisionsBtwnUpdates=2_steepness=1.0_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=${i}/study_df.pkl" --rl_algorithm_object_pickle="simulated_data/synthetic_mode=delayed_1_dosage_alg=sigmoid_LS_T=25_n=50_recruitN=50_decisionsBtwnUpdates=2_steepness=1.0_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=${i}/study_RLalg.pkl"
done




# python After_Study_Analyses.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --debug $debug --redo_analyses $redo_analyses --RL_alg $RL_alg --eta $eta --alg_state_feats $alg_state_feats --inference_mode $inference_mode --action_centering $action_centering
#kernprof -l After_Study_Analyses.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --debug $debug --RL_alg $RL_alg --eta $eta --alg_state_feats $alg_state_feats --inference_mode $inference_mode --action_centering $action_centering
#python -m line_profiler After_Study_Analyses.py.lprof
