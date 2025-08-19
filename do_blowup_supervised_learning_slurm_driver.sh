#!/bin/bash
#SBATCH --job-name=blowup_xgb
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4              # threads for OpenMP
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=/n/netscratch/murphy_lab/Lab/nclosser/blowup_supervised_learning/%A/results.out            # STDOUT
#SBATCH --error=/n/netscratch/murphy_lab/Lab/nclosser/blowup_supervised_learning/%A/results.out             # STDERR
#SBATCH --mail-type=END                                                                                     # We send an email when the job ends.
#SBATCH --mail-type=FAIL                                                                                    # We send an email when the job fails.
#SBATCH --mail-user=nowellclosser@g.harvard.edu                                                             # Email to which notifications will be sent

# Stop on nonzero exit codes and use of undefined variables
set -eu

# Load Python 3.10, among other things
echo $(date +"%Y-%m-%d %T") do_blowup_supervised_learning_slurm_driver.sh: Loading mamba and CUDA modules.
module load Mambaforge/22.11.1-fasrc01
module load cuda/12.2.0-fasrc01

# Make virtualenv if necessary, and then activate it
cd ~
if ! test -d venv; then
  echo $(date +"%Y-%m-%d %T") do_blowup_supervised_learning_slurm_driver.sh: Creating venv, as it did not exist.
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd ~/adaptive-sandwich
echo $(date +"%Y-%m-%d %T") do_blowup_supervised_learning_slurm_driver.sh: Making sure Python requirements are installed.
pip install -r requirements.txt
echo $(date +"%Y-%m-%d %T") do_blowup_supervised_learning_slurm_driver.sh: All Python requirements installed.

# Now run the Python script
echo $(date +"%Y-%m-%d %T") do_blowup_supervised_learning_slurm_driver.sh: Running do_blowup_supervised_learning.py.

python do_blowup_supervised_learning.py \
    --input_glob="$1" \
    --output_dir=/n/netscratch/murphy_lab/Lab/nclosser/blowup_supervised_learning/${SLURM_JOB_ID} \
    --empirical_trace_blowup_factor="$2" \
    --regression_label_type="$3" \
    --classification_label_type="$4"

echo $(date +"%Y-%m-%d %T") do_blowup_supervised_learning_slurm_driver.sh: Finished running do_blowup_supervised_learning.py. Exiting.