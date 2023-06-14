
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

Next
- Add action centering
    - Test out algorithm in practice
- Incremental recruitment


Next big things
- Code up new algorithm - make it all work with the finite differences grad
    - Test it out
- Incremental recruitment
- Anna's environment

`https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/src/rl_algorithm.py`

# https://github.com/pyutils/line_profiler

NEXT: get_weights function for sigmoid has data_dict argument that isn't needed, figure out collected_data_dict for smooth posterior sampling as well. get_weights
