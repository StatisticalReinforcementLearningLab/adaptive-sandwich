# Adaptive Sandwich Variance Package

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
- set PYTHONPATH to the root level of the module
- `./run.sh`, which outputs to `simulated_data/` by default. See all the possible flags to be toggled in the script code.

## Linting/Formatting

## TODO
1. Add precommit hooks (one of which should handle pip freeze)
2. Use class inheritance for RL algorithms?
3. Refactor After_Study_Analyses.py to not use global variables

