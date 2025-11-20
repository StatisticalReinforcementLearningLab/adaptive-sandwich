Comparing to the results of the following command on the branch `handle_action_centering_RL_dupe`:

`./run_local_synthetic.sh --T=10 --n=100 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --action_centering_inference=1 --recruit_n=20 --env_seed_override=1726458459 --alg_seed_override=1726463458`

The only difference between this test and the other similarly-named one is that one passes the loss function in and this one passes the estimating function in for RL (also maybe inference in the future).  The results should be approximately the same.