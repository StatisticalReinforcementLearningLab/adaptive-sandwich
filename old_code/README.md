
Make sure to use environment py39 to get scipy version '1.10.0'

4.10
- First thing is to get the autograd working for current algorithm
    - Write a function `get_stacked_estimating_function` that has product of weights and estimating equations for all parameters stacked

4.11
- Incorporate inference estimating equation into `get_stacked_estimating_function` 
- Form stacked meat estimator

4.12
- Match 'stacked bread' for finite differences
- Reproduce adaptive sandwich variance with finite differences

4.13
- Smooth posterior sampling basic way to get action selection probabilities
- Form action selection probabilities using estimating function parameters (inner function)

4.14
- Smooth posterior sampling estimating equations
- Smooth posterior samplign weights

4.15
- Code up smooth posterior sampling algorithm (no action centering)
- Line profiler to figure out what is taking so long...

4.17
- Add action centering for algorithm simulation (action selection and updates)

4.19
- Add action centering for after study analyses

6.1
- Sigmoid algorithm seems good
- data analysis: make it so you can break into chunks
- Working on smooth posterior sampling algorithm
    - timing: 18 seconds per run * 2000 runs / 60 sec / 60 min = 10 hours
    - timing action-centering: 40 seconds per run * 2000 runs / 60 sec / 60 min = 23 hours
- Incremental recruitment
    - Delayed updates seem okay for both boltzman and posterior sampling
    - Incremental recruitment on algorithm seems okay for sigmoid and posterior sampling
        - Something looks wrong with getting estimating equation in sigmoid algorithm... sum to zero check failing - estimating equation is outputting the wrong dimension matrix
    - **incremental recruitment on after study analyses** sigmoid and posterior sampling

Next
- Run algorithm in cloud

Challenges
- Timing - split up getting hessian finite differences

Next big things
- Code up new algorithm - make it all work with the finite differences grad
    - Test it out
- Incremental recruitment
- Integrate with Anna's code

`https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/src/rl_algorithm.py`

# https://github.com/pyutils/line_profiler

NEXT: get_weights function for sigmoid has data_dict argument that isn't needed, figure out collected_data_dict for smooth posterior sampling as well. get_weights




SAVE FROM ALGORITHM
- Algorithm dictionary
- Update dictionary - one for each update
    - user_id*
    - action*
    - prob1*
    - parameters*
    - base_states
    - treat_states
    - posterior_noise
- probabilities function
- estimating equation function 





Installation
- conda install numpy-indexed -c conda-forge







