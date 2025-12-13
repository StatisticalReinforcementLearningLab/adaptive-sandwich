```python
  _ _  __     _            _        _
 | (_)/ _|   (_)          | |      | |
 | |_| |_ ___ _  __ _  ___| | _____| |_
 | | |  _/ _ \ |/ _` |/ __| |/ / _ \ __|
 | | | ||  __/ | (_| | (__|   <  __/ |_
 |_|_|_| \___| |\__,_|\___|_|\_\___|\__|
            _/ |
           |__/
```

Save your standard errors from pooling in online decision-making algorithms.

## Setup (if not using conda)
### Create and activate a virtual environment
- `python3 -m venv .venv; source /.venv/bin/activate`

### Adding a package
- Add to `requirements.txt` with a specific version or no version if you want the latest stable
- Run `pip freeze > requirements.txt` to lock the versions of your package and all its subpackages

## Running the code
- `export PYTHONPATH to the absolute path of this repository on your computer
- `./run_local_synthetic.sh`, which outputs to `simulated_data/` by default. See all the possible flags to be toggled in the script code.

## Linting/Formatting

## Testing
python -m pytest
python -m pytest tests/unit_tests
python -m pytest tests/integration_tests


# Talk about gitignored cluster simulation scripts






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

