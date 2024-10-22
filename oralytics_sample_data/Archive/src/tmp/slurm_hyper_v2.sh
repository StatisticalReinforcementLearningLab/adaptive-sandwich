#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=5000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /Users/sghosh/Research/oralytics_algorithm_design/exps/write/hyper_v2/out_%j.txt # File to which STDOUT will be written
#SBATCH -e /Users/sghosh/Research/oralytics_algorithm_design/exps/write/hyper_v2/err_%j.txt # File to which STDERR will be written

python -u run.py /Users/sghosh/Research/oralytics_algorithm_design/exps/write/hyper_v2 hyper_v2 '{"sim_env_version": "v2", "base_env_type": "NON_STAT", "effect_size_scale": "small", "delayed_effect_scale": "LOW_R"}'
