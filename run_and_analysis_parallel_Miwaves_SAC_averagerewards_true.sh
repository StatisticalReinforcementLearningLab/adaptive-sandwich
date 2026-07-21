#!/bin/bash
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -t 0-23:20
#SBATCH --mem=64G
#SBATCH -p sapphire
#SBATCH -o /n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kesun@fas.harvard.edu

# Generate Monte Carlo data for the true MiWaves SAC average reward. The
# after-study sandwich analysis is intentionally skipped for large-n runs.
# Example:
# sbatch --array=0-999 run_and_analysis_parallel_Miwaves_SAC_averagerewards_true.sh --n=10000 --recruit_n=10000

set -eu

script_name="run_and_analysis_parallel_Miwaves_SAC_averagerewards_true.sh"
echo "$(date +"%Y-%m-%d %T") ${script_name}: Parsing options."

die() { echo "$*" >&2; exit 2; }
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# SAC hyperparameters; synchronized with run_local_Miwaves_SAC_averagerewards.sh.
ridge_penalty=10.0
epoch_actor=500
lr_pi=10.0
constant_ridge=0
actor_ridge_penalty=0.01

T=60
decisions_between_updates=1
update_cadence_offset=0
min_update_time=0
recruit_t=1
n=30
reward_modification=0

steepness=1.0
miwaves_habituation=1
miwaves_treatmenteffect=2

RL_alg="sac"
alg_state_feats="intercept,S1,S2,S3"
action_centering_RL=0
lclip=0.2
uclip=0.8
dynamic_seeds=0
env_seed_override=-1
alg_seed_override=-1
act_cost_threshold=0.1

# Long options use --name=value.
while getopts T:t:n:u:d:o:r:f:a:s:Y:A:G:F:L:M:w:x:v:W:V:-: OPT; do
  if [ "$OPT" = "-" ]; then
    OPT="${OPTARG%%=*}"
    OPTARG="${OPTARG#$OPT}"
    OPTARG="${OPTARG#=}"
  fi
  case "$OPT" in
    T  | max_time )                    needs_arg; T="$OPTARG" ;;
    t  | recruit_t )                   needs_arg; recruit_t="$OPTARG" ;;
    n  | num_users )                   needs_arg; n="$OPTARG" ;;
    u  | recruit_n )                   needs_arg; recruit_n="$OPTARG" ;;
    d  | decisions_between_updates )   needs_arg; decisions_between_updates="$OPTARG" ;;
    o  | update_cadence_offset )       needs_arg; update_cadence_offset="$OPTARG" ;;
    r  | RL_alg )                      needs_arg; RL_alg="$OPTARG" ;;
    f  | alg_state_feats )             needs_arg; alg_state_feats="$OPTARG" ;;
    a  | action_centering_RL )         needs_arg; action_centering_RL="$OPTARG" ;;
    s  | steepness )                   needs_arg; steepness="$OPTARG" ;;
    Y  | min_update_time )             needs_arg; min_update_time="$OPTARG" ;;
    A  | uclip )                       needs_arg; uclip="$OPTARG" ;;
    G  | lclip )                       needs_arg; lclip="$OPTARG" ;;
    F  | dynamic_seeds )               needs_arg; dynamic_seeds="$OPTARG" ;;
    L  | env_seed_override )           needs_arg; env_seed_override="$OPTARG" ;;
    M  | alg_seed_override )           needs_arg; alg_seed_override="$OPTARG" ;;
    miwaves_habituation )              needs_arg; miwaves_habituation="$OPTARG" ;;
    miwaves_treatmenteffect )          needs_arg; miwaves_treatmenteffect="$OPTARG" ;;
    reward_modification )              needs_arg; reward_modification="$OPTARG" ;;
    w  | ridge_penalty )               needs_arg; ridge_penalty="$OPTARG" ;;
    x  | epoch_actor )                 needs_arg; epoch_actor="$OPTARG" ;;
    v  | lr_pi )                       needs_arg; lr_pi="$OPTARG" ;;
    W  | constant_ridge )              needs_arg; constant_ridge="$OPTARG" ;;
    V  | actor_ridge_penalty )         needs_arg; actor_ridge_penalty="$OPTARG" ;;
    act_cost_threshold )               needs_arg; act_cost_threshold="$OPTARG" ;;
    \? )                               exit 2 ;;
    * )                                die "Illegal option --$OPT" ;;
  esac
done

shift $((OPTIND-1))
for arg in "$@"; do
  if [[ "$arg" != -* ]]; then
    die "Invalid argument: $arg. Options must start with a dash (- or --)."
  fi
done

if [ -z "${recruit_n:-}" ]; then
  recruit_n=$n
fi

echo "$(date +"%Y-%m-%d %T") ${script_name}: Loading inference_jax environment."
module load Mambaforge/22.11.1-fasrc01
mamba activate inference_jax
cd ~/2Longitudinal/adaptive-sandwich

save_dir_prefix="/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/Miwaves_sac_n${n}_T${T}"
save_dir="${save_dir_prefix}/${SLURM_ARRAY_TASK_ID}"
save_dir_glob="${save_dir_prefix}/*"
mkdir -p "$save_dir"

filename="_averagerewards"

echo "$(date +"%Y-%m-%d %T") ${script_name}: Beginning SAC data generation."
python rl_study_simulation_modified.py \
  --T="$T" \
  --N=1 \
  --parallel_task_index="$SLURM_ARRAY_TASK_ID" \
  --n="$n" \
  --dataset_type="miwaves" \
  --decisions_between_updates="$decisions_between_updates" \
  --update_cadence_offset="$update_cadence_offset" \
  --recruit_n="$recruit_n" \
  --recruit_t="$recruit_t" \
  --steepness="$steepness" \
  --RL_alg="$RL_alg" \
  --alg_state_feats="$alg_state_feats" \
  --action_centering="$action_centering_RL" \
  --dynamic_seeds="$dynamic_seeds" \
  --env_seed_override="$env_seed_override" \
  --alg_seed_override="$alg_seed_override" \
  --min_update_time="$min_update_time" \
  --upper_clip="$uclip" \
  --lower_clip="$lclip" \
  --save_dir="$save_dir" \
  --Twoarmed=0 \
  --filename="$filename" \
  --act_cost_threshold="$act_cost_threshold" \
  --Miwaves_habituation="$miwaves_habituation" \
  --Miwaves_treatmenteffect="$miwaves_treatmenteffect" \
  --reward_modification="$reward_modification" \
  --ridge_penalty="$ridge_penalty" \
  --epoch_actor="$epoch_actor" \
  --lr_pi="$lr_pi" \
  --constant_ridge="$constant_ridge" \
  --actor_ridge_penalty="$actor_ridge_penalty"

save_dir_suffix="decisionBtwnUpdates=${decisions_between_updates}_habituation=${miwaves_habituation}_treatment=${miwaves_treatmenteffect}_steepness=${steepness}_actorridge${actor_ridge_penalty}${filename}"
output_folder_glob="${save_dir_glob}/${save_dir_suffix}"

echo "$(date +"%Y-%m-%d %T") ${script_name}: Data generation complete."
echo "Generated data path pattern: ${output_folder_glob}/exp=1/data.csv"
