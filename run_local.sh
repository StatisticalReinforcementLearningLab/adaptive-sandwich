#!/bin/bash
set -eu

echo "$(date +"%Y-%m-%d %T") run.sh: Beginning simulation."

T=10
decisions_between_updates=2
# recruit_n=25; recruit_t=2
recruit_n=100; recruit_t=1
n=100
min_users=1
synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_1_dosage'
#synthetic_mode='delayed_01_5_dosage'
#synthetic_mode='test_1_1_T2'
#synthetic_mode='delayed_effects_large'
steepness=0.0
eta=0
RL_alg="sigmoid_LS"
#RL_alg="posterior_sampling"
#RL_alg="fixed_randomization"
#err_corr='independent'
err_corr='time_corr'
alg_state_feats="intercept,past_reward"
action_centering=0
#TODO: not used currently but maybe should be
debug=0
redo_analyses=1

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
    T  | max_time )                     needs_arg; T="$OPTARG" ;;
    t  | recruit_t )                    needs_arg; recruit_t="$OPTARG" ;;
    N  | num_simulations )              needs_arg; N="$OPTARG" ;;
    n  | num_users )                    needs_arg; n="$OPTARG" ;;
    u  | recruit_n )                    needs_arg; recruit_n="$OPTARG" ;;
    d  | decisions_between_updates )    needs_arg; decisions_between_updates="$OPTARG" ;;
    m  | min_users )                    needs_arg; min_users="$OPTARG" ;;
    r  | RL_alg )                       needs_arg; RL_alg="$OPTARG" ;;
    e  | err_corr )                     needs_arg; err_corr="$OPTARG" ;;
    f  | alg_state_feats )              needs_arg; alg_state_feats="$OPTARG" ;;
    a  | action_centering )             needs_arg; action_centering="$OPTARG" ;;
    s  | steepness )                    needs_arg; steepness="$OPTARG" ;;
    y  | synthetic_mode )               needs_arg; synthetic_mode="$OPTARG" ;;
    \? )                                exit 2 ;;  # bad short option (error reported via getopts)
    * )                                 die "Illegal option --$OPT" ;; # bad long option
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Simulate an RL study with the supplied arguments.  (We do just one repetition)
echo "$(date +"%Y-%m-%d %T") run.sh: Beginning RL study simulation."
python rl_study_simulation.py --T=$T --N=1 --n=$n --min_users=$min_users --decisions_between_updates $decisions_between_updates --recruit_n $recruit_n --recruit_t $recruit_t --synthetic_mode $synthetic_mode --steepness $steepness --RL_alg $RL_alg --err_corr $err_corr --alg_state_feats $alg_state_feats --action_centering $action_centering
echo "$(date +"%Y-%m-%d %T") run.sh: Finished RL study simulation."

# Create a convenience variable that holds the output folder for the last script
output_folder="simulated_data/synthetic_mode=${synthetic_mode}_alg=${RL_alg}_T=${T}_n=${n}_recruitN=${recruit_n}_decisionsBtwnUpdates=${decisions_between_updates}_steepness=${steepness}_algfeats=${alg_state_feats}_errcorr=${err_corr}_actionC=${action_centering}"

# Do after-study analysis on the single algorithm run from above
echo "$(date +"%Y-%m-%d %T") run.sh: Beginning after-study analysis."
python after_study_analysis.py analyze-dataset --study_dataframe_pickle="${output_folder}/exp=1/study_df.pkl" --rl_algorithm_object_pickle="${output_folder}/exp=1/study_RLalg.pkl"
echo "$(date +"%Y-%m-%d %T") run.sh: Ending after-study analysis."

echo "$(date +"%Y-%m-%d %T") run.sh: Finished simulation."
