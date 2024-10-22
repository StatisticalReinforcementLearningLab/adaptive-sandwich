#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=5000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /Users/sghosh/Research/oralytics_algorithm_design/exps/read/test_v3/out_%j.txt # File to which STDOUT will be written
#SBATCH -e /Users/sghosh/Research/oralytics_algorithm_design/exps/read/test_v3/err_%j.txt # File to which STDERR will be written

python -u run.py /Users/sghosh/Research/oralytics_algorithm_design/exps/read/test_v3 test_v3 '{"sim_env_version": "v3", "base_env_type": "NON_STAT", "effect_size_scale": "None", "delayed_effect_scale": "LOW_R", "alg_type": "BLR_AC_V3", "noise_var": "None", "clipping_vals": [0.2, 0.8], "b_logistic": 0.515, "update_cadence": 14, "cluster_size": "full_pooling", "cost_params": [80, 40]}'
