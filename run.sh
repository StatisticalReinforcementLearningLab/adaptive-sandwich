set -eu

N=10

n=50
#n=100
#n=500

#synthetic_mode='delayed_1_action_dosage'
# synthetic_mode='delayed_1_dosage_paper'
#synthetic_mode='delayed_5_action_dosage'
synthetic_mode='delayed_5_dosage_paper'

# steepness=0.5
# steepness=1
steepness=5

recruit_n=$n
T=50
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

python RL_Study_Simulation.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering

python After_Study_Analyses.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --debug $debug --RL_alg $RL_alg --eta $eta --alg_state_feats $alg_state_feats --inference_mode $inference_mode --action_centering $action_centering

#kernprof -l After_Study_Analyses.py --T=$T --N=$N --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --debug $debug --RL_alg $RL_alg --eta $eta --alg_state_feats $alg_state_feats --inference_mode $inference_mode --action_centering $action_centering

#python -m line_profiler After_Study_Analyses.py.lprof
