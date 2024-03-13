#!/bin/bash
#SBATCH -n 4                                      # Number of cores
#SBATCH -N 1                                      # Ensure that all cores are on one machine
#SBATCH -t 0-10:00                                # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                         # Partition to submit to
#SBATCH --mem=2G                                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm.%N.%j.out                        # STDOUT
#SBATCH -e slurm.%N.%j.err                        # STDERR
#SBATCH --mail-type=END                           # This command would send an email when the job ends.
#SBATCH --mail-type=FAIL                          # This command would send an email when the job ends.
#SBATCH --mail-user=nowellclosser@g.harvard.edu   # Email to which notifications will be sent

# Note that this script can be run interactively or with sbatch.  The above parameters should make sbatch
# get a GPU, and the below parameters can be used to change the simulation parameters. # If running
# interactively, one can start the session with something like the following:
# salloc -p gpu_test -t 0-02:00 --mem 16000 --gres=gpu:1 -n 8 -N 1

# Stop on nonzero exit codes and use of undefined variables, and print all commands
set -eu

echo $(date +"%Y-%m-%d %T") simulation_run.sh: Parsing options.

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts T:t:N:n:u:d:m:r:e:f:a:s:y:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    i  | input_glob )                 needs_arg; input_glob="$OPTARG" ;;
    \? )                                exit 2 ;;  # bad short option (error reported via getopts)
    * )                                 die "Illegal option --$OPT" ;; # bad long option
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

if [ -z "$input_glob" ]; then
        die 'Missing input folder arg'
fi

# Load Python 3.10, among other things
echo $(date +"%Y-%m-%d %T") simulation_run.sh: Loading mamba and CUDA modules.
module load Mambaforge/22.11.1-fasrc01
module load cuda/12.2.0-fasrc01

# Make virtualenv if necessary, and then activate it
cd ~
if ! test -d venv; then
  echo $(date +"%Y-%m-%d %T") simulation_run.sh: Creating venv, as it did not exist.
  python3 -m venv venv
fi
source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd ~/adaptive-sandwich
echo $(date +"%Y-%m-%d %T") simulation_run.sh: Making sure Python requirements are installed.
pip install -r simulation_requirements.txt
echo $(date +"%Y-%m-%d %T") simulation_run.sh: All Python requirements installed.

# Loop through each dataset created in the simulation (determined by number of Monte carlo repetitions)
# and do after-study analysis
echo $(date +"%Y-%m-%d %T") simulation_run.sh: Collecting pre-existing after-study analyses.
python after_study_analysis.py collect_existing_analyses --input_glob="${input_glob}"
echo $(date +"%Y-%m-%d %T") simulation_run.sh: Finished combining after-study analyses.

echo $(date +"%Y-%m-%d %T") simulation_run.sh: Simulation complete.