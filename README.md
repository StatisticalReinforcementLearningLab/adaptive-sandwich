# lifejacket: Save your standard errors from pooling in RL

- `After_Study_Analyses.py`: main code for after study analyses
- `RL_Study_Simulation.py`: main code for running RL algorithm experiements
- `basic_RL_algorithms.py`: includes code for boltzman sampling algorithm (called SigmoidLS)
- `debug_helper.py`: code I kept around from when I computed standard errors differently earlier. I keep it around to debug new code to make sure it matches.
- `helper_functions.py`: various helper functions
- `least_squares_helper.py`: functions to help with forming least squares estimators and estimating equations
- `oralytics_env.py`: oralytics environment (not cleaned up; haven't used in a while)
- `smooth_posterior_sampling.py`: code for smooth posterior sampling algorithm
- `synthetic_env.py`: synthetic simulation environment
- `run.sh`: script for running experiments


## Setup (if not using conda)
### Create and activate a virtual environment
- `python3 -m venv .venv; source /.venv/bin/activate`
### Adding a package
- Add to `requirements.txt` with a specific version or no version if you want the latest stable
- Run `pip freeze > requirements.txt` to lock the versions of your package and all its subpackages

## Running the code
- `export PYTHONPATH to the overall location of this repository on your computer
- `./run.sh`, which outputs to `simulated_data/` by default. See all the possible flags to be toggled in the script code.

## Linting/Formatting

## Testing
python -m pytest tests/unit_tests
python -m pytest tests/integration_tests







### Important Simulations (these are kind of like integration tests)

#### No adaptivity
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=50000 --steepness=0.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### No adaptivity, 5 batches incremental recruitment
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=10000 --steepness=0.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, no action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, no action_centering, 5 batches incremental recruitment
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=10000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### More adaptivity, no action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=50000 --steepness=5.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Even more adaptivity, no action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=50000 --steepness=10.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, RL action_centering, no inference action centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=1 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, inference action_centering, no RL action centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py" --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_action_centering.py"

#### Some adaptivity, inference and RL action_centering
sbatch --array=[0-999] -t 0-5:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=10 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, inference and RL action_centering, even more T
sbatch --array=[0-999] -t 1-00:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=25 --n=50000 --recruit_n=50000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"

#### Some adaptivity, inference and RL action_centering, even more T, 5 batches incremental recruitment
sbatch --array=[0-999] -t 1-00:00 --mem=50G simulation_run_and_analysis_parallel.sh --T=25 --n=50000 --recruit_n=10000 --steepness=3.0 --alg_state_feats=intercept,past_reward --action_centering_RL=0 --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py" --theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_no_action_centering.py"



## TODO
1. Add precommit hooks (one of which should handle pip freeze)

