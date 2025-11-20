# After study analysis for Linear Mixed Effects Bandits

## How to run
In order to run a simulation, please use the ```run_simulation.py``` file. It accepts command line parameters, specified in the file. This readme will be updated with examples

## Additional files
The following files are also provided to help the after-study analysis - mainly to be used with the after-study inference package:
- ```src/LME_action_selection.py```: This specifies the file with the ```LME_action_selection``` method, which is used to compute the action selection probabilities
- ```src/LME_estimating_function.py```: This specifies the file with the ```LME_estimating_function``` method, which is used to compute the estimating equations

## Output
The simulation will output 3 pickled data files in ```./results``` directory (unless explicity changed) for the after-study analysis:
- ```study_df.pkl```: The study dataframe
- ```estimating_equation_function_dict.pkl```: The parameters for the estimating equation function
- ```action_selection_function_dict.pkl```: The parameters for the action selection probability function

This repository also contains the file ```src/check_estimating_fn.py```, which takes in the path of the estimating equation parameter dictionary pickle file (as a global variable)
and outputs the sum of the estimating equation across users for each posterior update. These values should be near 0 if things are implemented correctly.
