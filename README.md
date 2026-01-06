





### Important Large-Scale Simulations

#### No adaptivity
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=50000 --steepness=0.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### No adaptivity, 5 batches incremental recruitment
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=10000 --steepness=0.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, no action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, no action_centering, 5 batches incremental recruitment
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=10000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### More adaptivity, no action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=50000 --steepness=5.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Even more adaptivity, no action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=50000 --steepness=10.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, RL action_centering, no inference action centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=1 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, inference action_centering, no RL action centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_action_centering.py"

#### Some adaptivity, inference and RL action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G run_and_analysis_parallel_synthetic --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=1 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_action_centering.py"

#### Some adaptivity, inference and RL action_centering, even more T
sbatch --array=[0-999] -t 1-00:00 --mem=50G run_and_analysis_parallel_synthetic --T=25 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=1 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, inference and RL action_centering, even more T, 5 batches incremental recruitment
sbatch --array=[0-999] -t 1-00:00 --mem=50G run_and_analysis_parallel_synthetic --T=25 --n=50000 --recruit_n=10000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=1 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"



## TODO
1. Add precommit hooks (pip freeze, linting, formatting)
2. Run tests on PRs

