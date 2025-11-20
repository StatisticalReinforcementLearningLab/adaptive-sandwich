
## Code for [_reBandit: Random Effects based Online RL algorithm for Reducing Cannabis Use_](https://arxiv.org/abs/2402.17739)

This code runs on **Python 3.9**.

### Setup Instructions

1. **Install Conda first**
2. Install the conda environment specified in `env.yml`

In general, our code requires the following Python packages:

- `numpy`
- `scipy`
- `sklearn`
- `pandas`
- `argparse`
- `matplotlib`
- **JAX** (most important)

> One can take advantage of **JAX with CUDA** to speed up runtime.

---

### Running the Simulation

Run `simulator_RD.py` with the following command-line arguments:

1. `-n` : Specify number of users to simulate (default: 120 users)  
2. `-d` : Specify number of days to simulate (default: 30 days, 60 decision points)  
3. `-s` : Seed to run (experiments used seeds from 0 to 499)  
4. `-rl` : The RL algorithm to run. Choose from: `random`, `BLR`, or `rebandit`  
5. `-tx` : Treatment effect setting. Choose from:
   - `overall_low` (Low TE)
   - `overall_high` (High TE)
   - `none` (Minimal TE)
6. `-dr` : Habituation factor (see Appendix A.8)
   - `-1` = no habituation
   - `1` = high habituation
   - `6` = low habituation (used in paper)
7. `-dp` : Proportion of population experiencing habituation
   - Set to `50` or `100` (as used in the paper)
8. `-act` : Lambda hyperparameter used in reward engineering  
   - Tried values: `[0, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2, 3]`

#### Example command

```bash
python simulator_RD.py -n 120 -d 30 -s 0 -rl "BLR" -tx "overall_low" -dr 6 -dp 50 -act 0.1
```

---

### Output

Simulation results are saved under:

```bash
./data/simulations/
```

Each simulation will be in a subfolder named by seed (e.g., `0`, `1`, …). Each seed folder contains:

- `simulation_output.csv` (main result)
- Other files (for logging/debugging)

> You can analyze these CSV files to replicate the results from the paper.

---

### Data Requirements and Disclaimer

- The simulator uses **SARA data**, which is **not publicly available**.
- A **template** CSV file (`combined_dataset.csv`) is provided – must be populated to use the simulator.
- Also provided: `MLR.pkl`, a pickle file of user models learned from SARA data.
