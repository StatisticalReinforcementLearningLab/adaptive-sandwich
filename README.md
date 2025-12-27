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

## Setup (if not using conda)
### Create and activate a virtual environment
- `python3 -m venv .venv; source /.venv/bin/activate`

### Adding a package
- Add to `requirements.txt` with a specific version or no version if you want the latest stable
- Run `pip freeze > requirements.txt` to lock the versions of your package and all its subpackages

## Running the code
- `export PYTHONPATH to the absolute path of this repository on your computer
- `./run_local_synthetic.sh`, which outputs to `simulated_data/` by default. See all the possible flags to be toggled in the script code.

## Linting/Formatting

## Testing
python -m pytest
python -m pytest tests/unit_tests
python -m pytest tests/integration_tests



## TODO
1. Add precommit hooks (pip freeze, linting, formatting)

