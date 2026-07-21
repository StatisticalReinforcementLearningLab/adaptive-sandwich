#!/bin/bash
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -t 0-23:20
#SBATCH --mem=64G
#SBATCH -p sapphire
#SBATCH -o /n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kesun@fas.harvard.edu

# Run one reproducible MiWaves SAC simulation and analysis per Slurm array task.
# Example:
# sbatch --array=0-999 run_and_analysis_parallel_Miwaves_SAC_averagerewards.sh

set -eu

script_name="run_and_analysis_parallel_Miwaves_SAC_averagerewards.sh"
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

# RL study configuration.
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

# Inference configuration.
in_study_col_name="in_study"
action_col_name="action"
policy_num_col_name="policy_num"
calendar_t_col_name="calendar_t"
user_id_col_name="user_id"
action_prob_col_name="action1prob"
reward_col_name="reward"
action_prob_func_filename="functions_to_pass_to_analysis/Miwaves_get_action_1_prob_SAC.py"
action_prob_func_args_beta_index=0
alg_update_func_filename="functions_to_pass_to_analysis/Miwaves_SAC_alg_update_function.py"
alg_update_func_type="estimating"
alg_update_func_args_beta_index=0
alg_update_func_args_action_prob_index=-1
alg_update_func_args_action_prob_times_index=-1
alg_update_func_args_previous_betas_index=1
inference_func_filename="functions_to_pass_to_analysis/primary_analysis_avg_reward_sum_loss.py"
inference_func_args_theta_index=0
inference_func_type="loss"
theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_avg_reward_sum.py"
# Slurm jobs cannot answer interactive prompts; all noninteractive checks remain enabled.
suppress_interactive_data_checks=1
suppress_all_data_checks=0
small_sample_correction="none"
collect_data_for_blowup_supervised_learning=0
stabilize_joint_adaptive_bread_inverse=0

# Long options use --name=value. MiWaves-specific names are long-only to avoid
# collisions with T/H/R, which already have established meanings.
while getopts T:t:n:u:d:o:r:f:a:s:Y:A:G:i:c:p:C:U:E:X:P:b:l:Z:B:D:j:I:h:g:F:L:M:Q:q:z:k:m:w:x:v:W:V:-: OPT; do
  if [ "$OPT" = "-" ]; then
    OPT="${OPTARG%%=*}"
    OPTARG="${OPTARG#$OPT}"
    OPTARG="${OPTARG#=}"
  fi
  case "$OPT" in
    T  | max_time )                                     needs_arg; T="$OPTARG" ;;
    t  | recruit_t )                                    needs_arg; recruit_t="$OPTARG" ;;
    n  | num_users )                                    needs_arg; n="$OPTARG" ;;
    u  | recruit_n )                                    needs_arg; recruit_n="$OPTARG" ;;
    d  | decisions_between_updates )                    needs_arg; decisions_between_updates="$OPTARG" ;;
    o  | update_cadence_offset )                        needs_arg; update_cadence_offset="$OPTARG" ;;
    r  | RL_alg )                                       needs_arg; RL_alg="$OPTARG" ;;
    f  | alg_state_feats )                              needs_arg; alg_state_feats="$OPTARG" ;;
    a  | action_centering_RL )                          needs_arg; action_centering_RL="$OPTARG" ;;
    s  | steepness )                                    needs_arg; steepness="$OPTARG" ;;
    Y  | min_update_time )                              needs_arg; min_update_time="$OPTARG" ;;
    A  | uclip )                                        needs_arg; uclip="$OPTARG" ;;
    G  | lclip )                                        needs_arg; lclip="$OPTARG" ;;
    i  | in_study_col_name )                            needs_arg; in_study_col_name="$OPTARG" ;;
    c  | action_col_name )                              needs_arg; action_col_name="$OPTARG" ;;
    p  | policy_num_col_name )                          needs_arg; policy_num_col_name="$OPTARG" ;;
    C  | calendar_t_col_name )                          needs_arg; calendar_t_col_name="$OPTARG" ;;
    U  | user_id_col_name )                             needs_arg; user_id_col_name="$OPTARG" ;;
    E  | action_prob_col_name )                         needs_arg; action_prob_col_name="$OPTARG" ;;
    X  | reward_col_name )                              needs_arg; reward_col_name="$OPTARG" ;;
    P  | action_prob_func_filename )                    needs_arg; action_prob_func_filename="$OPTARG" ;;
    b  | action_prob_func_args_beta_index )             needs_arg; action_prob_func_args_beta_index="$OPTARG" ;;
    l  | alg_update_func_filename )                     needs_arg; alg_update_func_filename="$OPTARG" ;;
    Z  | alg_update_func_type )                         needs_arg; alg_update_func_type="$OPTARG" ;;
    B  | alg_update_func_args_beta_index )              needs_arg; alg_update_func_args_beta_index="$OPTARG" ;;
    D  | alg_update_func_args_action_prob_index )       needs_arg; alg_update_func_args_action_prob_index="$OPTARG" ;;
    j  | alg_update_func_args_action_prob_times_index ) needs_arg; alg_update_func_args_action_prob_times_index="$OPTARG" ;;
    previous_betas_index | alg_update_func_args_previous_betas_index ) needs_arg; alg_update_func_args_previous_betas_index="$OPTARG" ;;
    I  | inference_func_filename )                      needs_arg; inference_func_filename="$OPTARG" ;;
    h  | inference_func_args_theta_index )              needs_arg; inference_func_args_theta_index="$OPTARG" ;;
    g  | inference_func_type )                          needs_arg; inference_func_type="$OPTARG" ;;
    theta_calculation_func_filename )                   needs_arg; theta_calculation_func_filename="$OPTARG" ;;
    F  | dynamic_seeds )                                needs_arg; dynamic_seeds="$OPTARG" ;;
    L  | env_seed_override )                            needs_arg; env_seed_override="$OPTARG" ;;
    M  | alg_seed_override )                            needs_arg; alg_seed_override="$OPTARG" ;;
    Q  | suppress_interactive_data_checks )             needs_arg; suppress_interactive_data_checks="$OPTARG" ;;
    q  | suppress_all_data_checks )                     needs_arg; suppress_all_data_checks="$OPTARG" ;;
    z  | small_sample_correction )                      needs_arg; small_sample_correction="$OPTARG" ;;
    k  | collect_data_for_blowup_supervised_learning )  needs_arg; collect_data_for_blowup_supervised_learning="$OPTARG" ;;
    m  | stabilize_joint_adaptive_bread_inverse )       needs_arg; stabilize_joint_adaptive_bread_inverse="$OPTARG" ;;
    miwaves_habituation )                               needs_arg; miwaves_habituation="$OPTARG" ;;
    miwaves_treatmenteffect )                           needs_arg; miwaves_treatmenteffect="$OPTARG" ;;
    reward_modification )                               needs_arg; reward_modification="$OPTARG" ;;
    act_cost_threshold )                                needs_arg; act_cost_threshold="$OPTARG" ;;
    w  | ridge_penalty )                                needs_arg; ridge_penalty="$OPTARG" ;;
    x  | epoch_actor )                                  needs_arg; epoch_actor="$OPTARG" ;;
    v  | lr_pi )                                        needs_arg; lr_pi="$OPTARG" ;;
    W  | constant_ridge )                               needs_arg; constant_ridge="$OPTARG" ;;
    V  | actor_ridge_penalty )                          needs_arg; actor_ridge_penalty="$OPTARG" ;;
    \? )                                                exit 2 ;;
    * )                                                 die "Illegal option --$OPT" ;;
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

echo "$(date +"%Y-%m-%d %T") ${script_name}: Beginning SAC simulation."
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

echo "$(date +"%Y-%m-%d %T") ${script_name}: Finished SAC simulation."

save_dir_suffix="decisionBtwnUpdates=${decisions_between_updates}_habituation=${miwaves_habituation}_treatment=${miwaves_treatmenteffect}_steepness=${steepness}_actorridge${actor_ridge_penalty}${filename}"
output_folder="${save_dir}/${save_dir_suffix}"
output_folder_glob="${save_dir_glob}/${save_dir_suffix}"

echo "$(date +"%Y-%m-%d %T") ${script_name}: Beginning after-study analysis."
lifejacket analyze \
  --study_df_pickle="${output_folder}/exp=1/study_df.pkl" \
  --action_prob_func_filename="$action_prob_func_filename" \
  --action_prob_func_args_pickle="${output_folder}/exp=1/pi_args.pkl" \
  --action_prob_func_args_beta_index="$action_prob_func_args_beta_index" \
  --alg_update_func_filename="$alg_update_func_filename" \
  --alg_update_func_type="$alg_update_func_type" \
  --alg_update_func_args_pickle="${output_folder}/exp=1/rl_update_args.pkl" \
  --alg_update_func_args_beta_index="$alg_update_func_args_beta_index" \
  --alg_update_func_args_action_prob_index="$alg_update_func_args_action_prob_index" \
  --alg_update_func_args_action_prob_times_index="$alg_update_func_args_action_prob_times_index" \
  --alg_update_func_args_previous_betas_index="$alg_update_func_args_previous_betas_index" \
  --inference_func_filename="$inference_func_filename" \
  --inference_func_args_theta_index="$inference_func_args_theta_index" \
  --inference_func_type="$inference_func_type" \
  --theta_calculation_func_filename="$theta_calculation_func_filename" \
  --in_study_col_name="$in_study_col_name" \
  --action_col_name="$action_col_name" \
  --policy_num_col_name="$policy_num_col_name" \
  --calendar_t_col_name="$calendar_t_col_name" \
  --user_id_col_name="$user_id_col_name" \
  --action_prob_col_name="$action_prob_col_name" \
  --reward_col_name="$reward_col_name" \
  --suppress_interactive_data_checks="$suppress_interactive_data_checks" \
  --suppress_all_data_checks="$suppress_all_data_checks" \
  --small_sample_correction="$small_sample_correction" \
  --collect_data_for_blowup_supervised_learning="$collect_data_for_blowup_supervised_learning" \
  --stabilize_joint_adaptive_bread_inverse="$stabilize_joint_adaptive_bread_inverse"

echo "$(date +"%Y-%m-%d %T") ${script_name}: Simulation and analysis complete."
echo "Collect with: bash simulation_collect_analyses.sh --input_glob=${output_folder_glob}/exp=1/analysis.pkl --num_users=$n --in_study_col_name=$in_study_col_name --action_col_name=$action_col_name --action_prob_col_name=$action_prob_col_name"
