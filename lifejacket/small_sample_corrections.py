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
    per_subject_joint_adjusted_meat_contributions,
    per_subject_classical_meat_contributions,
    per_subject_classical_bread_contributions,
    num_subjects,
    theta_dim,
):

    # We first compute the classical bread matrix and invert it.  While
    # it is possible to avoid this inversion using a QR decomposition and
    # solving linear systems (discussed more below), we typically don't have
    # issues with the conditioning of just the classical bread.
    classical_bread_matrix = jnp.mean(per_subject_classical_bread_contributions, axis=0)
    classical_bread_matrix = invert_matrix_and_check_conditioning(
        classical_bread_matrix,
    )[0]

    # These will hold either corrective matrices or scalar weights depending on
    # what small sample correction is requested.
    per_subject_adjusted_corrections = None
    per_subject_classical_corrections = None

    per_subject_adjusted_correction_weights = np.ones(num_subjects)
    per_subject_classical_correction_weights = np.ones(num_subjects)
    if small_sample_correction == SmallSampleCorrections.NONE:
        logger.info(
            "No small sample correction requested. Using the raw per-subject joint adjusted bread contributions."
        )
    elif small_sample_correction == SmallSampleCorrections.Z1theta:
        logger.info(
            "Using HC1 small sample correction at the subject trajectory level. Note that we are treating the number of parameters as simply the size of theta, despite the presence of betas."
        )
        per_subject_adjusted_correction_weights = (
            per_subject_classical_correction_weights
        ) = (num_subjects / (num_subjects - theta_dim) * np.ones(num_subjects))
    elif small_sample_correction in {
        SmallSampleCorrections.Z2theta,
        SmallSampleCorrections.Z3theta,
    }:
        logger.info("Using %s small sample correction at the subject trajectory level.")

        power = 1 if small_sample_correction == SmallSampleCorrections.Z2theta else 2

        # It turns out to typically not make sense to compute the adjusted analog
        # of the classical leverages, since this involves correcting the joint adjusted meat matrix
        # involving all beta and theta parameters.  HC2/HC3 corrections assume that
        # the number of parameters is smaller than the number of subjects, which will not typically be
        # the case if the number of subjects is small enough for these corrections to be important.
        # Therefore we also use the "classical" leverages for the adjusted correction weights, which
        # is sensible, corresponding to only adjusting based on the estimating equations for theta.

        # ALSO note that one way to test correctness of the leverages is that they should sum
        # to the number of inference parameters, ie the size of theta.  I tested that this is
        # true both for the classical leverages and the larger joint adjusted leverages when they
        # were still used, lending credence to the below calculations.

        # TODO: Write a unit test for some level of logic here and then rewrite this to not require
        # the classical bread explicitly. May be slower, probably needs a for loop so that can use
        # a solver for each matrix multiplication after a QR decomposition of the bread
        # transpose.
        classical_leverages_per_subject = (
            np.einsum(
                "nij,ji->n",
                per_subject_classical_bread_contributions,
                classical_bread_matrix,
            )
            / num_subjects
        )
        per_subject_classical_correction_weights = 1 / (
            (1 - classical_leverages_per_subject) ** power
        )

        per_subject_adjusted_correction_weights = (
            per_subject_classical_correction_weights
        )
    else:
        raise ValueError(
            f"Unknown small sample correction: {small_sample_correction}. "
            "Please choose from values in SmallSampleCorrections class."
        )

    # If we used matrix corrections, they will be stored as these corrections.
    # Otherwise, store the scalar weights.
    if per_subject_adjusted_corrections is None:
        per_subject_adjusted_corrections = per_subject_adjusted_correction_weights
    if per_subject_classical_corrections is None:
        per_subject_classical_corrections = per_subject_classical_correction_weights

    # The scalar corrections will have computed weights that need to be applied here,
    # whereas the matrix corrections will have been applied to the per-subject
    # contributions already.
    joint_adjusted_meat_matrix = jnp.mean(
        per_subject_adjusted_correction_weights[:, None, None]
        * per_subject_joint_adjusted_meat_contributions,
        axis=0,
    )
    classical_meat_matrix = jnp.mean(
        per_subject_classical_correction_weights[:, None, None]
        * per_subject_classical_meat_contributions,
        axis=0,
    )

    return (
        joint_adjusted_meat_matrix,
        classical_meat_matrix,
        per_subject_adjusted_corrections,
        per_subject_classical_corrections,
    )
