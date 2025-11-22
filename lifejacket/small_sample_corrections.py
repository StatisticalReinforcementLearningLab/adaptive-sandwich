import logging

import numpy as np
from jax import numpy as jnp

from .constants import SmallSampleCorrections
from .helper_functions import invert_matrix_and_check_conditioning

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def perform_desired_small_sample_correction(
    small_sample_correction,
    per_user_joint_adaptive_meat_contributions,
    per_user_classical_meat_contributions,
    per_user_classical_bread_inverse_contributions,
    num_users,
    theta_dim,
):

    # We first compute the classical inverse bread matrix and invert it.  While
    # it is possible to avoid this inversion using a QR decomposition and
    # solving linear systems (discussed more below), we typically don't have
    # issues with the conditioning of just the classical bread.
    classical_bread_inverse_matrix = jnp.mean(
        per_user_classical_bread_inverse_contributions, axis=0
    )
    classical_bread_matrix = invert_matrix_and_check_conditioning(
        classical_bread_inverse_matrix,
    )[0]

    # These will hold either corrective matrices or scalar weights depending on
    # what small sample correction is requested.
    per_user_adaptive_corrections = None
    per_user_classical_corrections = None

    per_user_adaptive_correction_weights = np.ones(num_users)
    per_user_classical_correction_weights = np.ones(num_users)
    if small_sample_correction == SmallSampleCorrections.NONE:
        logger.info(
            "No small sample correction requested. Using the raw per-user joint adaptive bread inverse contributions."
        )
    elif small_sample_correction == SmallSampleCorrections.HC1theta:
        logger.info(
            "Using HC1 small sample correction at the user trajectory level. Note that we are treating the number of parameters as simply the size of theta, despite the presence of betas."
        )
        per_user_adaptive_correction_weights = per_user_classical_correction_weights = (
            num_users / (num_users - theta_dim) * np.ones(num_users)
        )
    elif small_sample_correction in {
        SmallSampleCorrections.HC2theta,
        SmallSampleCorrections.HC3theta,
    }:
        logger.info("Using %s small sample correction at the user trajectory level.")

        power = 1 if small_sample_correction == SmallSampleCorrections.HC2theta else 2

        # It turns out to typically not make sense to compute the adaptive analog
        # of the classical leverages, since this involves correcting the joint adaptive meat matrix
        # involving all beta and theta parameters.  HC2/HC3 corrections assume that
        # the number of parameters is smaller than the number of users, which will not typically be
        # the case if the number of users is small enough for these corrections to be important.
        # Therefore we also use the "classical" leverages for the adaptive correction weights, which
        # is sensible, corresponding to only adjusting based on the estimating equations for theta.

        # ALSO note that one way to test correctness of the leverages is that they should sum
        # to the number of inference parameters, ie the size of theta.  I tested that this is
        # true both for the classical leverages and the larger joint adaptive leverages when they
        # were still used, lending credence to the below calculations.

        # TODO: Write a unit test for some level of logic here and then rewrite this to not require
        # the classical bread explicitly. May be slower, probably needs a for loop so that can use
        # a solver for each matrix multiplication after a QR decomposition of the bread inverse
        # transpose.
        classical_leverages_per_user = (
            np.einsum(
                "nij,ji->n",
                per_user_classical_bread_inverse_contributions,
                classical_bread_matrix,
            )
            / num_users
        )
        per_user_classical_correction_weights = 1 / (
            (1 - classical_leverages_per_user) ** power
        )

        per_user_adaptive_correction_weights = per_user_classical_correction_weights
    else:
        raise ValueError(
            f"Unknown small sample correction: {small_sample_correction}. "
            "Please choose from values in SmallSampleCorrections class."
        )

    # If we used matrix corrections, they will be stored as these corrections.
    # Otherwise, store the scalar weights.
    if per_user_adaptive_corrections is None:
        per_user_adaptive_corrections = per_user_adaptive_correction_weights
    if per_user_classical_corrections is None:
        per_user_classical_corrections = per_user_classical_correction_weights

    # The scalar corrections will have computed weights that need to be applied here,
    # whereas the matrix corrections will have been applied to the per-user
    # contributions already.
    joint_adaptive_meat_matrix = jnp.mean(
        per_user_adaptive_correction_weights[:, None, None]
        * per_user_joint_adaptive_meat_contributions,
        axis=0,
    )
    classical_meat_matrix = jnp.mean(
        per_user_classical_correction_weights[:, None, None]
        * per_user_classical_meat_contributions,
        axis=0,
    )

    return (
        joint_adaptive_meat_matrix,
        classical_meat_matrix,
        per_user_adaptive_corrections,
        per_user_classical_corrections,
    )
