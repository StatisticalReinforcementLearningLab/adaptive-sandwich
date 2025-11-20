# Stop on nonzero exit codes and use of undefined variables, and print all commands
set -eu

echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: Parsing options.

die() { echo "$*" >&2; exit 2; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

input_glob=""
num_users=""
index_to_check_ci_coverage=""
study_df_filename="study_df.pkl"

# Parse single-char options as directly supported by getopts, but allow long-form
# under - option.  The :'s signify that arguments are required for these options.
while getopts i:n:c:s:a:p:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    i  | input_glob )                   needs_arg; input_glob="$OPTARG" ;;
    n  | num_users )                    needs_arg; num_users="$OPTARG" ;;
    c  | index_to_check_ci_coverage )   needs_arg; index_to_check_ci_coverage="$OPTARG" ;;
    s  | in_study_col_name )            needs_arg; in_study_col_name="$OPTARG" ;;
    a  | action_col_name )              needs_arg; action_col_name="$OPTARG" ;;
    p  | action_prob_col_name )         needs_arg; action_prob_col_name="$OPTARG" ;;
    f  | study_df_filename )            needs_arg; study_df_filename="$OPTARG" ;;
    \? )                                exit 2 ;;  # bad short option (error reported via getopts)
    * )                                 die "Illegal option --$OPT" ;; # bad long option
  esac
done

shift $((OPTIND-1)) # remove parsed options and args from $@ list

# Check for invalid options that do not start with a dash. This
# prevents accidentally missing dashes and thinking you passed an
# arg that you didn't.
for arg in "$@"; do
  if [[ "$arg" != -* ]]; then
    die "Invalid argument: $arg. Options must start with a dash (- or --)."
  fi
done

if [ -z "$input_glob" ]; then
  die 'Missing input folder arg'
fi

# Load Python 3.10, among other things
echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: Loading mamba module.
# module load Mambaforge/22.11.1-fasrc01

# Make virtualenv if necessary, and then activate it
# cd ~
# if ! test -d venv; then
#   echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: Creating venv, as it did not exist.
#   python3 -m venv venv
# fi
# source venv/bin/activate

# Now install all Python requirements.  This is incremental, so it's ok to do every time.
cd ~/2Longitudinal/adaptive-sandwich
echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: Making sure Python requirements are installed.
# pip install -r requirements.txt
echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: All Python requirements installed.

# Loop through each dataset created in the referenced simulation and do
# after-study analysis
echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: Collecting pre-existing after-study analyses.
if [ -z "$index_to_check_ci_coverage" ]; then
  python after_study_analysis_partial.py collect-existing-analyses --input_glob="${input_glob}" --num_users="${num_users}" --study_df_filename="${study_df_filename}"
else
  python after_study_analysis_partial.py collect-existing-analyses --input_glob="${input_glob}" --num_users="${num_users}" --study_df_filename="${study_df_filename}" --index_to_check_ci_coverage="${index_to_check_ci_coverage}" --in_study_col_name=$in_study_col_name --action_col_name=$action_col_name --action_prob_col_name=$action_prob_col_name
fi

echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: Finished combining after-study analyses.

echo $(date +"%Y-%m-%d %T") simulation_collect_analyses.sh: Analysis complete.