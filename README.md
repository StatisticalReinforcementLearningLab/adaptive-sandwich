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

### Current Progress (updated by Ke in 01/06/2026)

> SAC on Synthetic enviroment: although the updated lifejacket package can handle the RL algorithm with recursive updates, the variance estimate from our adaptive approach becomes much large such that the coverage rate is close to 1. Empirical results are provided in Section 3 in ``ke/Algorithm_SAC.tex``.

> Tompson Sampling on Miwaves enviroment: Under our adaptive method, inference after TS works well in the synthetic environment, but it leads to much larger variance estimate in the Miwaves environment (we increase the sample size to 300). I am still checking whether TS can achive a reasonable policy learning in this environment to look into the reason behind. Empirical results are provided in Section 1.1 in ``ke/Algorithm_TS.tex``.

> SAC on Miwaves enviroment [TODO]


### Main Usage Instruction 

The codebase in this branch is adapted from the main branch by adding new inference objectives(``ke/Inference_treatmenteffect.tex``), Soft Actor Critic algorithm(``ke/Algorithm_SAC.tex``), and Miwaves environment([rebandit paper](https://arxiv.org/abs/2402.17739)). The tex files mentioned here refers to those in the overleaf project ``Inference After Pooling Susan Version``.

Specifically, we can change the environment and choose the RL algorithms based on our research need to optimize the intervention and collect the data. The function of afterstudy inference is then calleds to analyze the collected dataset by using the seperate ``lifejacket`` package, which is mostly unable to be modified. The current implementation have included two inference objectives, two enviroments, and two RL algorithms:

> Inference objectives: average rewards, treatment effect in a two-armed trials

> Environment: Synthetic one (with dosage variable), Miwaves (based on the [Miwaves simulator](https://github.com/susobhang70/reBandit)) 

> Online RL algorithms: 

### Step 1: pull the code from Ke branch

``git pull origin ke``

### Step 2: download the packages

``pip install -r requirements.txt``

Most importantly, the up-to-dated lifejacket from the main branch will be installed (pip install directly) that works both in the local laptop or the cluster. Sometimes, if we need to modify the lifejacket, we can use ``add_general_beta_differentiation_to_RL_side`` branch for testing. If so, we can install the lifejacket package via ``-e git+https://github.com/StatisticalReinforcementLearningLab/adaptive-sandwich.git@add_general_beta_differentiation_to_RL_side#egg=lifejacket`` in the ``requirement.txt``.


### Step 3: local testing

> ``run_local_syn_TS_averagerewards.sh``: synthetic environment, Thompson sampling, average rewards for inference

> ``run_local_syn_TS_treatmenteffect.sh``: synthetic environment, Thompson sampling, treatment effect in a two-armed trial for inference

> ``run_local_syn_SAC_averagerewards.sh``: synthetic environment, SAC, average rewards for inference

> ``run_local_syn_SAC_treatmenteffect.sh``: synthetic environment, SAC, treatment effect in a two-armed trial for inference

> ``run_local_Miwaves_TS_averagerewards.sh``: Miwaves environment, Thompson sampling, average rewards for inference

> ``run_local_Miwaves_[TS/SAC]_[averagerewards/treatmenteffect].sh``: [TODO]


The main purpose of local testing is to check the nearly zero estimating function and increasing trend in the learning curves. Hare are caveat of each new component:

> Treatment effect in a two-armed trial: we use a modified synthetic environment by incorporating some pre-treated features in the reward generation process. Please refer to Section 1.2 in ``ke/Inference_treatmenteffect.tex``.

> SAC algorithm: The implementation of SAC is harder than TS as we need to perform gradient descent to update the actor to achieve a nearly zero estimating equations, which can cause some instability for the after study inference. Please refer to Eq.8 of Section 1 in ``ke/Algorithm_SAC.tex``.

> Miwave environment: For synthetic enviroment, we use ``rl_study_simulation_partial.py``, while for Miwaves environment, we use ``rl_study_simulation_partial_Miwaves.py``


### Step 4: Overall evaluation on the cluster across multiple replications

Once we have passed the local test, we can create a virtual enviroment in the clustser, pip install the packages needed, and then perform a final evaluation across multiple replications in the cluster.

> ``run_and_analysis_parallel_syn_TS_averagerewards.sh``: synthetic environment, Thompson sampling, average rewards for inference

> ``run_and_analysis_parallel_syn_TS_treatmenteffect.sh``: synthetic environment, Thompson sampling, treatment effect in a two-armed trial for inference

> ``run_and_analysis_parallel_syn_SAC_averagerewards.sh``: synthetic environment, SAC, average rewards for inference

> ``run_and_analysis_parallel_syn_SAC_treatmenteffect.sh``: synthetic environment, SAC, treatment effect in a two-armed trial for inference

> ``run_and_analysis_parallel_Miwaves_TS_averagerewards.sh``: Miwaves environment, Thompson sampling, average rewards for inference

> ``run_and_analysis_parallel_Miwaves_[TS/SAC]_[averagerewards/treatmenteffect].sh``: [TODO]

The running examples can be:

```
sbatch --array=[0-999] -n 16 -t 0-23:59 -p serial_requeue --mem=64G run_and_analysis_parallel_syn_SAC_averagerewards.sh -T 50 -n 30  --steepness=1.0 --synthetic_mode='delayed_1_action_dosage' --lclip=0.1 --uclip=0.9 
sbatch --array=[0-999] -t 0-23:59 -p serial_requeue --mem=64G run_and_analysis_parallel_Miwaves_TS_averagerewards.sh -T 60 -n 30 --decisions_between_updates=1
```


After the running is complete, we can use the evaluation script to calculate the variance estimate and coverage rate. For example, in Miwaves, we can run:

```
bash simulation_collect_analyses.sh --input_glob=/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/n500_T60/*/simulated_data/miwaves_alg=smooth_posterior_sampling_T=60_n=500_decisionsBtwnUpdates=1_actionC=0_averagerewards/exp=1/analysis.pkl --num_users=500 --index_to_check_ci_coverage=0 --in_study_col_name=in_study --action_col_name=action --action_prob_col_name=action1prob
```

