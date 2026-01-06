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


### Overview

This branch (`ke`) extends the main codebase by adding:
- new inference objectives (`ke/Inference_treatmenteffect.tex`),
- an online Soft Actor-Critic (SAC) algorithm (`ke/Algorithm_SAC.tex`),
- and the Miwaves environment ([reBandit paper](https://arxiv.org/abs/2402.17739)).

The TeX files referenced here correspond to the Overleaf project *Inference After Pooling (Susan Version)*.

In this branch, we vary environments and online RL algorithms to optimize interventions and collect adaptive data. After data collection, after-study inference is performed using the separate `lifejacket` package, which is treated as a largely fixed dependency. The current implementation includes two inference objectives, two environments, and two online RL algorithms:

- **Inference objectives:**
  - average rewards
  - treatment effect in a two-armed trial

- **Environment:** 
  - Synthetic one (with dosage variable), 
  - Miwaves (based on the [Miwaves simulator](https://github.com/susobhang70/reBandit)) 

- **Online RL algorithms:**
  - Thompson Sampling
  - Soft Actor Critic 

### Step 1: pull the code from Ke branch

```
git pull origin ke
```


### Step 2: download the packages

```
pip install -r requirements.txt
```

This branch relies on the up-to-date `lifejacket` package from the main branch, which supports recursive updates required by online RL algorithms.

- By default, `lifejacket` is installed via pip.
- For development or testing purposes, one may instead install the experimental branch ``add_general_beta_differentiation_to_RL_side`` for testing by adding ``-e git+https://github.com/StatisticalReinforcementLearningLab/adaptive-sandwich.git@add_general_beta_differentiation_to_RL_side#egg=lifejacket`` in the ``requirements.txt``.


### Step 3: local testing

- ``run_local_syn_TS_averagerewards.sh``: synthetic environment, Thompson sampling, average rewards for inference

- ``run_local_syn_TS_treatmenteffect.sh``: synthetic environment, Thompson sampling, treatment effect in a two-armed trial for inference

- ``run_local_syn_SAC_averagerewards.sh``: synthetic environment, SAC, average rewards for inference

- ``run_local_syn_SAC_treatmenteffect.sh``: synthetic environment, SAC, treatment effect in a two-armed trial for inference

- ``run_local_Miwaves_TS_averagerewards.sh``: Miwaves environment, Thompson sampling, average rewards for inference

- ``run_local_Miwaves_[TS/SAC]_[averagerewards/treatmenteffect].sh``: [TODO]


Local testing serves as a sanity check for near-zero estimating equations and monotonic improvement in learning curves. Below are important notes for each component:

- **Treatment effect in a two-armed trial**: we use a modified synthetic environment by incorporating some pre-treated features in the reward generation process. Please refer to Section 1.2 in ``ke/Inference_treatmenteffect.tex``.

- **SAC algorithm**: The implementation of SAC is harder than TS as we need to perform gradient descent to update the actor to achieve a nearly zero estimating equations, which can cause some instability for the after study inference. Please refer to Eq.8 of Section 1 in ``ke/Algorithm_SAC.tex``.

- **Miwave environment**: For synthetic environment, we use ``rl_study_simulation_partial.py``, while for Miwaves environment, we use ``rl_study_simulation_partial_Miwaves.py``. The two files can be unified in the future.


### Step 4: Overall evaluation on the cluster across multiple replications

Once we have passed the local test, we can create a virtual environment in the cluster, pip install the packages needed, and then perform a final evaluation across multiple replications in the cluster.

- ``run_and_analysis_parallel_syn_TS_averagerewards.sh``: synthetic environment, Thompson sampling, average rewards for inference

- ``run_and_analysis_parallel_syn_TS_treatmenteffect.sh``: synthetic environment, Thompson sampling, treatment effect in a two-armed trial for inference

- ``run_and_analysis_parallel_syn_SAC_averagerewards.sh``: synthetic environment, SAC, average rewards for inference

- ``run_and_analysis_parallel_syn_SAC_treatmenteffect.sh``: synthetic environment, SAC, treatment effect in a two-armed trial for inference

- ``run_and_analysis_parallel_Miwaves_TS_averagerewards.sh``: Miwaves environment, Thompson sampling, average rewards for inference

- ``run_and_analysis_parallel_Miwaves_[TS/SAC]_[averagerewards/treatmenteffect].sh``: [TODO]

Examples of the running command can be:

> synthetic environment + SAC + average rewards


```
sbatch --array=[0-999] -n 16 -t 0-23:59 -p serial_requeue --mem=64G run_and_analysis_parallel_syn_SAC_averagerewards.sh -T 50 -n 30  --steepness=1.0 --synthetic_mode='delayed_1_action_dosage' --lclip=0.1 --uclip=0.9 
```

> Miwaves environment + TS + average rewards

```
sbatch --array=[0-999] -t 0-23:59 -p serial_requeue --mem=64G run_and_analysis_parallel_Miwaves_TS_averagerewards.sh -T 60 -n 30 --decisions_between_updates=1
````


After the running is complete, we can use the evaluation script to calculate the variance estimate and coverage rate. The command can be found from the output log. For example, in Miwaves, we can run:

```
bash simulation_collect_analyses.sh --input_glob=/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/n500_T60/*/simulated_data/miwaves_alg=smooth_posterior_sampling_T=60_n=500_decisionsBtwnUpdates=1_actionC=0_averagerewards/exp=1/analysis.pkl --num_users=500 --index_to_check_ci_coverage=0 --in_study_col_name=in_study --action_col_name=action --action_prob_col_name=action1prob
```

One needs to replace ``kesun`` by their folder name and be careful about the file path. Hare are some caveat:

- Unlike the original implementation in the main branch, the directory path for the saved data in the bash files in this branch is slightly changed, which facilitates the varying sample size n in the experiments but may cause inconsistency.

- table_results_*.py: these files are used to save the evaluation results

## Development Notes (Last updated by Ke on 01/06/2026)

- SAC on Synthetic environment: although the updated lifejacket package can handle the RL algorithm with recursive updates, the variance estimate from our adaptive approach becomes overly large such that the coverage rate is close to 1. Empirical results are provided in Section 3 in ``ke/Algorithm_SAC.tex``.

- Thompson Sampling on Miwaves environment: Under our adaptive method, inference after TS works well in the synthetic environment, but it leads to much larger variance estimate in the Miwaves environment (we increase the sample size to 300). We are checking whether TS can achieve a reasonable policy learning in this environment to look into the reason behind. Empirical results are provided in Section 1.1 in ``ke/Algorithm_TS.tex``.

- SAC on Miwaves environment [TODO]

