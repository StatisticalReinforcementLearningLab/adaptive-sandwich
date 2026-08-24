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
- `export PYTHONPATH to the absolute path of this repository on your computer
- `./run_local_synthetic.sh`, which outputs to `simulated_data/` by default. See all the possible flags to be toggled in the script code.

## Linting/Formatting

## Testing
uv run python -m pytest
uv run python -m pytest tests/unit_tests
uv run python -m pytest tests/integration_tests



## TODO
1. Add pre-commit hooks (linting, formatting)

