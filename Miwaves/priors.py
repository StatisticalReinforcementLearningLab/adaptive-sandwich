# Setting the priors

import numpy as np
from scipy import linalg

BASELINE_PRIOR_MEAN = np.array([2.12, 0.00, 0.0, -0.69, 0.0, 0.0, 0.0, 0.0])
BASELINE_PRIOR_VAR = np.diag(
    [
        (0.78**2),
        (0.38**2),
        (0.62**2),
        (0.98**2),
        (0.16**2),
        (0.1**2),
        (0.16**2),
        (0.1**2),
    ]
)
ADVANTAGE_PRIOR_MEAN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ADVANTAGE_PRIOR_VAR = np.diag(
    [
        (0.27**2),
        (0.33**2),
        (0.3**2),
        (0.32**2),
        (0.1**2),
        (0.1**2),
        (0.1**2),
        (0.1**2),
    ]
)

PRIOR_MEAN = {}
PRIOR_VAR = {}
INIT_COV = {}

PRIOR_MEAN[0] = np.hstack(
    (BASELINE_PRIOR_MEAN, ADVANTAGE_PRIOR_MEAN, ADVANTAGE_PRIOR_MEAN)
)
PRIOR_VAR[0] = linalg.block_diag(
    BASELINE_PRIOR_VAR, ADVANTAGE_PRIOR_VAR, ADVANTAGE_PRIOR_VAR
)

INIT_COV[0] = np.diag([0.01 for i in range(len(PRIOR_MEAN[0]))])


PRIOR_MEAN[1] = np.hstack(
    (BASELINE_PRIOR_MEAN[:4], ADVANTAGE_PRIOR_MEAN[:4], ADVANTAGE_PRIOR_MEAN[:4])
)
PRIOR_VAR[1] = linalg.block_diag(
    BASELINE_PRIOR_VAR[:4, :4], ADVANTAGE_PRIOR_VAR[:4, :4], ADVANTAGE_PRIOR_VAR[:4, :4]
)

INIT_COV[1] = np.diag([0.01 for i in range(len(PRIOR_MEAN[1]))])


PRIOR_MEAN[2] = np.hstack(
    (BASELINE_PRIOR_MEAN[:4], ADVANTAGE_PRIOR_MEAN[:1], ADVANTAGE_PRIOR_MEAN[:1])
)
PRIOR_VAR[2] = linalg.block_diag(
    BASELINE_PRIOR_VAR[:4, :4], ADVANTAGE_PRIOR_VAR[:1, :1], ADVANTAGE_PRIOR_VAR[:1, :1]
)

INIT_COV[2] = np.diag([0.01 for i in range(len(PRIOR_MEAN[2]))])


PARAM_SIZE = [[8, 8, 8], [4, 4, 4], [4, 1, 1]]

PRIOR_NOISE_VAR = 0.85