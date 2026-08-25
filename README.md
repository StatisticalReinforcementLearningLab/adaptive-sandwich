```python
  _ _  __     _            _        _
 | (_)/ _|   (_)          | |      | |
 | |_| |_ ___ _  __ _  ___| | _____| |_
 | | |  _/ _ \ |/ _` |/ __| |/ / _ \ __|
 | | | ||  __/ | (_| | (__|   <  __/ |_
 |_|_|_| \___| |\__,_|\___|_|\_\___|\__|
            _/ |
           |__/
```

Save your standard errors from pooling in online decision-making algorithms.

## Setup
This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
- Install uv (see the [install docs](https://docs.astral.sh/uv/getting-started/installation/))
- Run `uv sync --extra dev` to create `.venv` and install locked dependencies
- Prefix commands with `uv run`, e.g. `uv run python -m pytest`, or `source .venv/bin/activate` to activate the environment directly

### Adding a package
- Runtime dependency: `uv add <package>`
- Dev/test-only dependency: `uv add --optional dev <package>`
- Either updates `pyproject.toml` and `uv.lock` together

## Running the code

### From the command line
The primary interface is the `lifejacket analyze` CLI command (installed by `uv sync`). It takes pickled inputs (an analysis dataframe plus the per-subject/per-decision-time argument dictionaries for your action-probability, algorithm-update, and inference functions) and paths to the Python files defining those functions:

```bash
uv run lifejacket analyze \
  --analysis_df_pickle=study_df.pkl \
  --action_prob_func_filename=functions_to_pass_to_analysis/synthetic_get_action_1_prob_generalized_logistic.py \
  --action_prob_func_args_pickle=pi_args.pkl \
  --action_prob_func_args_beta_index=0 \
  --alg_update_func_filename=functions_to_pass_to_analysis/RL_least_squares_loss_regularized.py \
  --alg_update_func_type=loss \
  --alg_update_func_args_pickle=rl_update_args.pkl \
  --alg_update_func_args_beta_index=0 \
  --inference_func_filename=functions_to_pass_to_analysis/synthetic_get_least_squares_loss_inference_no_action_centering.py \
  --inference_func_type=loss \
  --inference_func_args_theta_index=0 \
  --theta_calculation_func_filename=functions_to_pass_to_analysis/synthetic_estimate_theta_least_squares_no_action_centering.py \
  --active_col_name=in_study \
  --action_col_name=action \
  --policy_num_col_name=policy_num \
  --calendar_t_col_name=calendar_t \
  --subject_id_col_name=user_id \
  --action_prob_col_name=action1prob \
  --reward_col_name=reward
```

Run `uv run lifejacket analyze --help` for the full option reference (every
option's docstring, and which ones are required vs. optional). To try the
whole pipeline end-to-end on synthetic data without supplying your own
dataset, run `run_local_synthetic.sh` from *within*
`tests/simulators_and_runners/` (it resolves its own Python scripts relative
to the current directory, not the repo root) — this is exactly the command
above, wired up with generated pickles:

    cd tests/simulators_and_runners && ./run_local_synthetic.sh

This simulates a study and then calls `lifejacket analyze` on it, outputting
to `tests/simulators_and_runners/simulated_data/` by default (gitignored).
With default settings it will interactively ask you to confirm that skipping
small-sample corrections is intentional; answer `y`, or pass
`--suppress_interactive_data_checks=1` to skip all such prompts. See the
script itself for all the flags it accepts.

### From Python
`lifejacket analyze` is a thin wrapper around
`lifejacket.post_deployment_analysis.analyze_dataset`, which takes the exact
same information as in-memory Python objects instead of file paths — useful
if you're already building these objects in a script or notebook. Given the
pickles and function files `run_local_synthetic.sh` produces (see above), the
equivalent direct call is:

```python
import pickle
from lifejacket.helper_functions import load_function_from_same_named_file
from lifejacket.post_deployment_analysis import analyze_dataset

exp_dir = "tests/simulators_and_runners/simulated_data/<scenario>/exp=1"
with open(f"{exp_dir}/study_df.pkl", "rb") as f:
    analysis_df = pickle.load(f)
with open(f"{exp_dir}/pi_args.pkl", "rb") as f:
    action_prob_func_args = pickle.load(f)
with open(f"{exp_dir}/rl_update_args.pkl", "rb") as f:
    alg_update_func_args = pickle.load(f)

analyze_dataset(
    output_dir=exp_dir,
    analysis_df=analysis_df,
    action_prob_func=load_function_from_same_named_file(
        "functions_to_pass_to_analysis/synthetic_get_action_1_prob_generalized_logistic.py"
    ),
    action_prob_func_args=action_prob_func_args,
    action_prob_func_args_beta_index=0,
    alg_update_func=load_function_from_same_named_file(
        "functions_to_pass_to_analysis/RL_least_squares_loss_regularized.py"
    ),
    alg_update_func_type="loss",
    alg_update_func_args=alg_update_func_args,
    alg_update_func_args_beta_index=0,
    alg_update_func_args_action_prob_index=5,
    alg_update_func_args_action_prob_times_index=6,
    alg_update_func_args_previous_betas_index=-1,
    inference_func=load_function_from_same_named_file(
        "functions_to_pass_to_analysis/synthetic_get_least_squares_loss_inference_no_action_centering.py"
    ),
    inference_func_type="loss",
    inference_func_args_theta_index=0,
    theta_calculation_func=load_function_from_same_named_file(
        "functions_to_pass_to_analysis/synthetic_estimate_theta_least_squares_no_action_centering.py"
    ),
    active_col_name="in_study",
    action_col_name="action",
    policy_num_col_name="policy_num",
    calendar_t_col_name="calendar_t",
    subject_id_col_name="user_id",
    action_prob_col_name="action1prob",
    reward_col_name="reward",
    suppress_interactive_data_checks=True,
    suppress_all_data_checks=False,
    small_sample_correction="none",
    collect_data_for_blowup_supervised_learning=False,
    form_adjusted_meat_adjustments_explicitly=False,
    stabilize_joint_bread=False,
)
```

The two argument dictionaries (`action_prob_func_args`, `alg_update_func_args`)
are the least obvious pieces if you're building them yourself rather than
loading them from a prior run: they're nested per-decision-time/per-update
dictionaries keyed by subject id, matching each function's own argument
tuple. See `tests/unit_tests/test_post_deployment_analysis.py`'s `setup_data_*`
fixtures for worked examples of constructing them from scratch, or
`tests/simulators_and_runners/rl_study_simulation.py` for how the synthetic
simulator builds them.

## Linting/Formatting
This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, run automatically via [pre-commit](https://pre-commit.com/) before every commit.

### Install the hooks (once per clone)

```bash
uv run pre-commit install
```

### Run the hooks manually on all files

```bash
uv run pre-commit run --all-files
```

Hooks configured in `.pre-commit-config.yaml`:

| Hook | What it does |
|------|-------------|
| `ruff` | Lint with auto-fix |
| `ruff-format` | Enforce consistent formatting |
| `pytest-unit` | Run `tests/unit_tests` (excluding `slow`-marked tests) before every commit |

## Testing
uv run python -m pytest
uv run python -m pytest tests/unit_tests
uv run python -m pytest tests/integration_tests

### Performance benchmarks
`tests/benchmarks` is a phase-timing + numerical-regression benchmark for
`analyze_dataset`, separate from correctness testing, at two scales ("small",
always fast; "medium", matching `tests/integration_tests`' scale, marked
`slow`). Run with `-s` to see the per-phase wall-clock breakdown:

    python -m pytest tests/benchmarks -v -s              # both scales
    python -m pytest tests/benchmarks -v -s -m "not slow" # small only, fast

See `docs/adr/0001-adaptive-sandwich-performance-plan.md` for why this exists
(including a JIT compile-time regression it caught) and
`tests/benchmarks/generate_fixture.py` for how to regenerate its fixtures.

