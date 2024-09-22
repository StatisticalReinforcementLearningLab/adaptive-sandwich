import pickle
import os
import logging
import warnings
import pathlib
import glob

import click
import numpy as np
from jax import numpy as jnp
import scipy

import calculate_derivatives
import input_checks


from helper_functions import (
    invert_matrix_and_check_conditioning,
    load_module_from_source_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# TODO: Break this file up


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--input_glob",
    help="A glob that captures all of the analyses to be collected.  Leaf folders will be searched for analyses",
    required=True,
)
def collect_existing_analyses(input_glob):

    theta_estimates = []
    adaptive_sandwich_var_estimates = []
    classical_sandwich_var_estimates = []
    filenames = glob.glob(input_glob)

    logger.info("Found %d files under the glob %s", len(filenames), input_glob)
    if len(filenames) == 0:
        raise RuntimeError("Aborting because no files found. Please check path.")

    for i, filename in enumerate(filenames):
        if i and i % (len(filenames) // 10) == 0:
            logger.info("A(nother) tenth of files processed.")
        if not os.stat(filename).st_size:
            raise RuntimeError(
                "Empty analysis pickle.  This means there were probably timeouts or other failures during simulations."
            )
        with open(filename, "rb") as f:
            analysis_dict = pickle.load(f)
            (
                theta_est,
                adaptive_sandwich_var,
                classical_sandwich_var,
            ) = (
                analysis_dict["theta_est"],
                analysis_dict["adaptive_sandwich_var_estimate"],
                analysis_dict["classical_sandwich_var_estimate"],
            )
            theta_estimates.append(theta_est)
            adaptive_sandwich_var_estimates.append(adaptive_sandwich_var)
            classical_sandwich_var_estimates.append(classical_sandwich_var)

    theta_estimates = np.array(theta_estimates)
    adaptive_sandwich_var_estimates = np.array(adaptive_sandwich_var_estimates)
    classical_sandwich_var_estimates = np.array(classical_sandwich_var_estimates)

    theta_estimate = np.mean(theta_estimates, axis=0)
    empirical_var_normalized = empirical_var_normalized = np.atleast_2d(
        np.cov(theta_estimates.T, ddof=0)
    )
    mean_adaptive_sandwich_var_estimate = np.mean(
        adaptive_sandwich_var_estimates, axis=0
    )
    mean_classical_sandwich_var_estimate = np.mean(
        classical_sandwich_var_estimates, axis=0
    )

    # Calculate standard error (or corresponding variance) of variance estimate for each
    # component of theta.  This is done by finding an unbiased estimator of the standard
    # formula for the standard error of a variance from iid observations.
    # Population standard error formula: https://en.wikipedia.org/wiki/Variance
    # Unbiased estimator: https://stats.stackexchange.com/questions/307537/unbiased-estimator-of-the-variance-of-the-sample-variance
    theta_component_variance_std_errors = []
    for i in range(len(theta_estimate)):
        component_estimates = [estimate[i] for estimate in theta_estimates]
        second_central_moment = scipy.stats.moment(component_estimates, moment=4)
        fourth_central_moment = scipy.stats.moment(component_estimates, moment=4)
        n = len(theta_estimates)
        theta_component_variance_std_errors.append(
            np.sqrt(
                n
                * (
                    ((n) ** 2 - 3) * (second_central_moment) ** 2
                    + ((n - 1) ** 2) * fourth_central_moment
                )
                / ((n - 3) * (n - 2) * ((n - 1) ** 2))
            )
        )

    approximate_standard_errors = np.empty_like(empirical_var_normalized)
    for i, j in np.ndindex(approximate_standard_errors.shape):
        approximate_standard_errors[i, j] = max(
            theta_component_variance_std_errors[i],
            theta_component_variance_std_errors[j],
        )

    print(f"\nParameter estimate:\n{theta_estimate}")
    print(f"\nEmpirical variance:\n{empirical_var_normalized}")
    print(
        f"\nEmpirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):\n{approximate_standard_errors}"
    )
    print(
        f"\nAdaptive sandwich variance estimate:\n{mean_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\nClassical sandwich variance estimate:\n{mean_classical_sandwich_var_estimate}\n",
    )
    print(
        f"\nAdaptive sandwich variance estimate std errors from empirical:\n{(mean_adaptive_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}",
    )
    print(
        f"\nClassical sandwich variance estimate std errors from empirical:\n{(mean_classical_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}\n",
    )


# TODO: Add option to give per-user loss OR estimating function. Just loss now
# TODO: take in theta instead of forming it, and use to check estimating function.
#       Yet we still want to support large simulation case where we DO calculate theta.
#       I think you have to pass in theta-forming function OR theta itself.
# TODO: Take in requirements files for action prob and loss and take derivatives
# in corresponding sandbox. For now we just assume the dependencies in this package
# suffice.
# TODO: Handle raw timestamps instead of calendar time index? For now I'm requiring it.
# More generally, handle decision times being different across different users? Would like
# to consolidate.
# TODO: Check all help strings for accuracy.
# TODO: Don't use theta and beta jargon?? Need a legend if I do.
# TODO: Make run scripts that hardcode to action centering or not on both RL and inference sides
# TODO: Need to support pure exploration phase with more flags than just in study. Maybe in study, receiving updates
# TODO: Deal with NA, -1, etc policy numbers
@cli.command()
@click.option(
    "--study_df_pickle",
    type=click.File("rb"),
    help="Pickled pandas dataframe in correct format (see contract/readme)",
    required=True,
)
@click.option(
    "--action_prob_func_filename",
    type=click.Path(exists=True),
    help="File that contains the action probability function and relevant imports.  The filename will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--action_prob_func_args_pickle",
    type=click.File("rb"),
    help="Pickled dictionary that contains the action probability function arguments for all decision times for all users",
    required=True,
)
@click.option(
    "--action_prob_func_args_beta_index",
    type=int,
    required=True,
    help="Index of the RL parameter vector beta in the tuple of action probability func args",
)
@click.option(
    "--rl_loss_func_filename",
    type=click.Path(exists=True),
    help="File that contains the per-user loss function used to determine the RL parameters at each update and relevant imports.  The filename will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--rl_loss_func_args_pickle",
    type=click.File("rb"),
    help="Pickled dictionary that contains the RL loss function arguments for all update times for all users",
    required=True,
)
@click.option(
    "--rl_loss_func_args_beta_index",
    type=int,
    required=True,
    help="Index of the RL parameter vector beta in the tuple of RL loss func args",
)
@click.option(
    "--rl_loss_func_args_action_prob_index",
    type=int,
    default=-1000,
    help="Index of the action probability in the tuple of RL loss func args, if applicable",
)
@click.option(
    "--rl_loss_func_args_action_prob_times_index",
    type=int,
    default=-1000,
    help="Index of the argument holding the decision times the action probabilities correspond to in the tuple of RL loss func args, if applicable",
)
@click.option(
    "--inference_loss_func_filename",
    type=click.Path(exists=True),
    help="File that contains the per-user loss function used to determine the inference estimate and relevant imports.  The filename will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--inference_loss_func_args_theta_index",
    type=int,
    required=True,
    help="Index of the RL parameter vector beta in the tuple of inference loss func args",
)
@click.option(
    "--theta_calculation_func_filename",
    type=click.Path(exists=True),
    help="File that allows one to actually calculate a theta estimate given the study dataframe only. One must supply either this or a precomputed theta estimate.",
    required=True,
)
@click.option(
    "--in_study_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the study dataframe that indicates whether a user is in the study",
)
@click.option(
    "--action_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the study dataframe that indicates which action was taken",
)
@click.option(
    "--policy_num_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates the policy number in use",
)
@click.option(
    "--calendar_t_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates calendar time (shared integer index across users).",
)
@click.option(
    "--user_id_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates user id",
)
@click.option(
    "--action_prob_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that gives action probabilities",
)
def analyze_dataset(
    study_df_pickle,
    action_prob_func_filename,
    action_prob_func_args_pickle,
    action_prob_func_args_beta_index,
    rl_loss_func_filename,
    rl_loss_func_args_pickle,
    rl_loss_func_args_beta_index,
    rl_loss_func_args_action_prob_index,
    rl_loss_func_args_action_prob_times_index,
    inference_loss_func_filename,
    inference_loss_func_args_theta_index,
    theta_calculation_func_filename,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_col_name,
):
    """

    I check estimating function sum 0 later, but differentiate between
    RL and beta side? Also could move check here. Might be nice to have all
    checks in same place. Wherever that check is, should allow user to see
    and verify close enough to zero

    Make sure in study is never on for more than one stretch EDIT: unclear if
    this will remain an invariant as we deal with more complicated data missingness

    Possibly make sure no real-looking data when in_study is off

    I think I'm agnostic to indexing of calendar times but should check because
    otherwise need to add a check here to verify required format.

    Currently assuming function args can be placed in a numpy array. Must be scalar, 1d or 2d array.
    Higher dimensional objects not supported.  Not entirely sure what kind of "scalars" apply.

    Should be clear from dataframe spec but beta must be vector (not matrix)

    Codify assumptions that make get_first_applicable_time work.  The main
    thing is an assumption that users don't get different policies at the same
    time.  EDIT: Well... users can have different policies at the same time. So
    we can't codify this and have to rewrite that function.

    Codify assumptions used for collect_batched_in_study_actions

    Can we check rl loss and policy arg Falsiness against study df availability indicators?

    Make the user give the min and max probabilities, and I'll enforce it

    Flag to toggle interactive checks, if any

    I assume someone is in the study at each decision time. Check for this or
    see if shouldn't always be true. EDIT: Is this true?

    I also assume someone has some data to contribute at each update time. Check
    for this or see if shouldn't always be true. EDIT: Is this true?

    Should I have an explicit check for theta func args in study df instead of letting
    fail?
    """
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    study_df = pickle.load(study_df_pickle)
    # TODO: Should I sort? Check how slow it is, for one.
    # study_df = pickle.load(study_df_pickle).sort_values(
    #     by=[user_id_col_name, calendar_t_col_name]
    # )
    action_prob_func_args = pickle.load(action_prob_func_args_pickle)
    rl_loss_func_args = pickle.load(rl_loss_func_args_pickle)

    beta_dim = calculate_beta_dim(rl_loss_func_args, rl_loss_func_args_beta_index)

    theta_est = estimate_theta(study_df, theta_calculation_func_filename)

    # This does the first round of input validation, before computing any
    # gradients
    input_checks.perform_first_wave_input_checks(
        study_df,
        in_study_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        user_id_col_name,
        action_prob_col_name,
        action_prob_func_filename,
        action_prob_func_args,
        action_prob_func_args_beta_index,
        rl_loss_func_args,
        rl_loss_func_args_beta_index,
        rl_loss_func_args_action_prob_index,
        rl_loss_func_args_action_prob_times_index,
        theta_est,
    )

    algorithm_statistics_by_calendar_t = calculate_algorithm_statistics(
        study_df,
        in_study_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        user_id_col_name,
        action_prob_func_filename,
        action_prob_func_args,
        action_prob_func_args_beta_index,
        rl_loss_func_filename,
        rl_loss_func_args,
        rl_loss_func_args_beta_index,
        rl_loss_func_args_action_prob_index,
        rl_loss_func_args_action_prob_times_index,
    )
    upper_left_bread_inverse = calculate_upper_left_bread_inverse(
        study_df, user_id_col_name, beta_dim, algorithm_statistics_by_calendar_t
    )

    (
        adaptive_sandwich_var_estimate,
        classical_sandwich_var_estimate,
        joint_bread_inverse_matrix,
        joint_meat_matrix,
        inference_loss_gradients,
        inference_loss_hessians,
        inference_loss_gradient_pi_derivatives,
    ) = compute_variance_estimates(
        study_df,
        beta_dim,
        theta_est,
        algorithm_statistics_by_calendar_t,
        upper_left_bread_inverse,
        inference_loss_func_filename,
        inference_loss_func_args_theta_index,
        in_study_col_name,
        user_id_col_name,
        action_prob_col_name,
        calendar_t_col_name,
    )

    # Write analysis results to same directory that input files are in
    folder_path = pathlib.Path(study_df_pickle.name).parent.resolve()
    with open(f"{folder_path}/analysis.pkl", "wb") as f:
        pickle.dump(
            {
                "theta_est": theta_est,
                "adaptive_sandwich_var_estimate": adaptive_sandwich_var_estimate,
                "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
            },
            f,
        )

    with open(f"{folder_path}/debug_pieces.pkl", "wb") as f:
        pickle.dump(
            {
                "theta_est": theta_est,
                "adaptive_sandwich_var_estimate": adaptive_sandwich_var_estimate,
                "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
                "joint_bread_inverse_matrix": joint_bread_inverse_matrix,
                "joint_meat_matrix": joint_meat_matrix,
                "inference_loss_gradients": inference_loss_gradients,
                "inference_loss_hessians": inference_loss_hessians,
                "inference_loss_gradient_pi_derivatives": inference_loss_gradient_pi_derivatives,
                "algorithm_statistics_by_calendar_t": algorithm_statistics_by_calendar_t,
                "upper_left_bread_inverse": upper_left_bread_inverse,
            },
            f,
        )

    print(f"\nParameter estimate:\n {theta_est}")
    print(f"\nAdaptive sandwich variance estimate:\n {adaptive_sandwich_var_estimate}")
    print(
        f"\nClassical sandwich variance estimate:\n {classical_sandwich_var_estimate}\n"
    )


def calculate_beta_dim(rl_loss_func_args, rl_loss_func_args_beta_index):
    for user_args_dict in rl_loss_func_args.values():
        for args in user_args_dict.values():
            if args:
                return args[rl_loss_func_args_beta_index].size


# TODO: Docstring
def estimate_theta(study_df, theta_calculation_func_filename):
    logger.info("Forming theta estimate.")
    # Retrieve the RL function from file
    theta_calculation_module = load_module_from_source_file(
        "theta_calculation", theta_calculation_func_filename
    )
    # NOTE the assumption that the function and file have the same name
    theta_calculation_func_name = os.path.basename(
        theta_calculation_func_filename
    ).split(".")[0]
    try:
        theta_calculation_func = getattr(
            theta_calculation_module, theta_calculation_func_name
        )
    except AttributeError as e:
        raise ValueError(
            "Unable to import theta estimation function.  Please verify the file has the same name as the function of interest."
        ) from e

    return theta_calculation_func(study_df)


# TODO: docstring
def calculate_algorithm_statistics(
    study_df,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_func_filename,
    action_prob_func_args,
    action_prob_func_args_beta_index,
    rl_loss_func_filename,
    rl_loss_func_args,
    rl_loss_func_args_beta_index,
    rl_loss_func_args_action_prob_index,
    rl_loss_func_args_action_prob_times_index,
):
    pi_and_weight_gradients_by_calendar_t = (
        calculate_derivatives.calculate_pi_and_weight_gradients(
            study_df,
            in_study_col_name,
            action_col_name,
            calendar_t_col_name,
            user_id_col_name,
            action_prob_func_filename,
            action_prob_func_args,
            action_prob_func_args_beta_index,
        )
    )
    rl_update_derivatives_by_calendar_t = (
        calculate_derivatives.calculate_rl_loss_derivatives(
            study_df,
            rl_loss_func_filename,
            rl_loss_func_args,
            rl_loss_func_args_beta_index,
            rl_loss_func_args_action_prob_index,
            rl_loss_func_args_action_prob_times_index,
            policy_num_col_name,
            calendar_t_col_name,
        )
    )

    merged_dict = {}
    for t, t_dict in pi_and_weight_gradients_by_calendar_t.items():
        merged_dict[t] = {
            **t_dict,
            **rl_update_derivatives_by_calendar_t.get(t, {}),
        }

    return merged_dict


# TODO: docstring
# TODO: One of the hotspots for update time logic to be removed
def calculate_upper_left_bread_inverse(
    study_df, user_id_col_name, beta_dim, algorithm_statistics_by_calendar_t
):

    # List of times that were the first applicable time for some update
    # TODO: sort to not rely on insertion order?
    # TODO: use policy_num in df? alg statistics potentially ok too though.
    next_times_after_update = [
        t
        for t, value in algorithm_statistics_by_calendar_t.items()
        if "loss_gradients_by_user_id" in value
    ]

    # Form the dimensions for our bread matrix portion (pre-inverting)
    num_updates = len(next_times_after_update)
    overall_dim = beta_dim * num_updates
    output_matrix = jnp.zeros((overall_dim, overall_dim))

    user_ids = study_df[user_id_col_name].unique()
    num_users = len(user_ids)

    # This simply collects the pi derivatives with respect to betas for all
    # decision times for each user. The one complication is that we add some
    # padding of zeros for decision times before the first update to make
    # indexing simpler below.
    # NOTE there was a bug here that ASSUMED the padding needed to happen,
    # in particular that the algo statistics started at
    # next_times_after_update[0].  This is not necessarily true, and is now
    # dictated by the args passed to us.  Because I want to allow the user to
    # pass pi args for all decision times (in fact this should be the default),
    # I instead will make this the time I deal with that. I will just zero out
    # any pi gradients until after the first update.  Note that isn't necessary;
    # we could do nothing, because this is just about getting the right values
    # at the right index.  But then we are assuming that we have pi gradients
    # from the beginning.  Instead just take this heavy-handed approach and
    # ensure we have the shape we want whether data starts immediately after or
    # sometime before the first update.
    # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
    pi_derivatives_by_user_id = {
        user_id: jnp.pad(
            jnp.array(
                [
                    t_dict["pi_gradients_by_user_id"][user_id]
                    for t, t_dict in algorithm_statistics_by_calendar_t.items()
                    if t >= next_times_after_update[0]
                ]
            ),
            pad_width=((next_times_after_update[0] - 1, 0), (0, 0)),
        )
        for user_id in user_ids
    }

    # This loop iterates over all times that were the first applicable time
    # for a non-initial policy. Take care to note that update_idx starts at 0.
    # Think of each iteration of this loop as creating a (block) row of the matrix
    for update_idx, next_t_after_update in enumerate(next_times_after_update):
        logger.info(
            "Processing the update that first applied at time %s.", next_t_after_update
        )
        t_stats_dict = algorithm_statistics_by_calendar_t[next_t_after_update]

        # This loop creates the non-diagonal terms for the current update
        # Think of each iteration of this loop as creating one term in the current (block) row
        logger.info("Creating the non-diagonal terms for the current update.")
        for i in range(update_idx):
            lower_t = next_times_after_update[i]
            upper_t = next_times_after_update[i + 1]
            running_entry_holder = jnp.zeros((beta_dim, beta_dim))

            # This loop calculates the per-user quantities that will be
            # averaged for the final matrix entries
            for user_id, loss_gradient in t_stats_dict[
                "loss_gradients_by_user_id"
            ].items():
                weight_gradient_sum = jnp.zeros(beta_dim)

                # This loop iterates over decision times in slices
                # according to what was used for each update to collect the
                # right weight gradients
                for t in range(
                    lower_t,
                    upper_t,
                ):
                    weight_gradient_sum += algorithm_statistics_by_calendar_t[t][
                        "weight_gradients_by_user_id"
                    ][user_id]

                running_entry_holder += jnp.outer(
                    loss_gradient,
                    weight_gradient_sum,
                )

                # TODO: Detailed comment explaining this logic and the data
                # orientation that makes it work.  Also note the assumption
                # that the estimating function is additive across times
                # so that matrix multiplication is the right operation. Also
                # place this comment on the after study analysis logic or
                # link to the same explanation in both places.
                # Maybe link to a document with a picture...
                # TODO: This assumes indexing starts at 1
                mixed_theta_beta_loss_derivative = jnp.matmul(
                    t_stats_dict["loss_gradient_pi_derivatives_by_user_id"][user_id][
                        :,
                        lower_t - 1 : upper_t - 1,
                    ],
                    pi_derivatives_by_user_id[user_id][
                        lower_t - 1 : upper_t - 1,
                        :,
                    ],
                )
                running_entry_holder += mixed_theta_beta_loss_derivative
            # TODO: Use jnp.block instead of indexing
            output_matrix = output_matrix.at[
                (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
                i * beta_dim : (i + 1) * beta_dim,
            ].set(running_entry_holder / num_users)

        # Add the diagonal hessian entry (which is already an average)
        # TODO: Use jnp.block instead of indexing
        output_matrix = output_matrix.at[
            (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
            (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
        ].set(t_stats_dict["avg_loss_hessian"])

    return output_matrix


# TODO: Docstring
# TODO: One of the hotspots for update time logic to be removed
def compute_variance_estimates(
    study_df,
    beta_dim,
    theta_est,
    algorithm_statistics_by_calendar_t,
    upper_left_bread_inverse,
    inference_loss_func_filename,
    inference_loss_func_args_theta_index,
    in_study_col_name,
    user_id_col_name,
    action_prob_col_name,
    calendar_t_col_name,
):
    # List of times that were the first applicable time for some update
    # Sorting shouldn't be necessary, as insertion order should be chronological
    # but we do it just in case.
    update_times = sorted(
        [
            t
            for t, value in algorithm_statistics_by_calendar_t.items()
            if "loss_gradients_by_user_id" in value
        ]
    )

    # Collect list of user ids to guarantee we have a shared, fixed order
    # to iterate through in a variety of places.
    user_ids = study_df[user_id_col_name].unique()

    logger.info("Forming adaptive sandwich variance estimator.")

    logger.info("Calculating all derivatives needed with JAX")
    loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
        calculate_derivatives.calculate_inference_loss_derivatives(
            study_df,
            theta_est,
            inference_loss_func_filename,
            inference_loss_func_args_theta_index,
            user_ids,
            user_id_col_name,
            action_prob_col_name,
            in_study_col_name,
            calendar_t_col_name,
        )
    )

    logger.info("Forming adaptive bread inverse and inverting.")
    max_t = study_df[calendar_t_col_name].max()
    theta_dim = len(theta_est)
    joint_bread_inverse_matrix = form_bread_inverse_matrix(
        upper_left_bread_inverse,
        max_t,
        algorithm_statistics_by_calendar_t,
        update_times,
        beta_dim,
        theta_dim,
        user_ids,
        loss_gradients,
        loss_hessians,
        loss_gradient_pi_derivatives,
    )
    logger.info("Adaptive joint bread inverse:")
    logger.info(joint_bread_inverse_matrix)
    joint_bread_matrix = invert_matrix_and_check_conditioning(
        joint_bread_inverse_matrix
    )

    logger.info("Forming adaptive meat.")
    # TODO: Small sample corrections
    joint_meat_matrix = form_meat_matrix(
        theta_dim,
        update_times,
        beta_dim,
        algorithm_statistics_by_calendar_t,
        user_ids,
        loss_gradients,
    )
    logger.info("Adaptive joint meat:")
    logger.info(joint_meat_matrix)

    logger.info("Combining sandwich ingredients.")
    # Note the normalization here: underlying the calculations we have asymptotic normality
    # at rate sqrt(n), so in finite samples we approximate the observed variance of theta itself
    # by dividing the variance of that limiting normal by a factor of n.  This is happening in the
    # behind the scenes in the classical function as well.
    joint_adaptive_variance = (
        joint_bread_matrix @ joint_meat_matrix @ joint_bread_matrix.T
    ) / len(user_ids)
    logger.info("Finished forming adaptive sandwich variance estimator.")

    return (
        # These are what's actually needed
        joint_adaptive_variance[-len(theta_est) :, -len(theta_est) :],
        get_classical_sandwich_var(theta_dim, loss_gradients, loss_hessians),
        # These are returned for debugging purposes
        joint_bread_inverse_matrix,
        joint_meat_matrix,
        loss_gradients,
        loss_hessians,
        loss_gradient_pi_derivatives,
    )


# TODO: doc string
def form_meat_matrix(
    theta_dim, update_times, beta_dim, algo_stats_dict, user_ids, loss_gradients
):
    num_rows_cols = beta_dim * len(update_times) + theta_dim
    # TODO: Why do I do this type conversion?
    running_meat_matrix = jnp.zeros((num_rows_cols, num_rows_cols)).astype(jnp.float32)
    estimating_function_sum = jnp.zeros((num_rows_cols, 1))
    fallback_beta_gradient = jnp.zeros(beta_dim)

    for i, user_id in enumerate(user_ids):
        user_meat_vector = jnp.concatenate(
            # beta estimating functions
            # TODO: Pretty sure this should have zeros when not in study but verify
            # TODO: Should arguably set things up so fallback isn't needed (0 gradients
            # for everyone not in study) but maybe good to have regardless. Could
            # also add a check that all update times have gradients for all users.
            [
                algo_stats_dict[t]["loss_gradients_by_user_id"].get(
                    user_id, fallback_beta_gradient
                )
                for t in update_times
            ]
            # theta estimating function
            + [loss_gradients[i]],
        ).reshape(-1, 1)
        running_meat_matrix += jnp.outer(user_meat_vector, user_meat_vector)
        estimating_function_sum += user_meat_vector

    # TODO: The check for the beta gradients should probably be upstream at the
    # time of reformatting the RL data in the intermediate stage. Also we may want
    # this to be more than a warning eventually.
    if not jnp.allclose(
        estimating_function_sum,
        jnp.zeros((num_rows_cols, 1)),
    ):
        warnings.warn(
            f"Estimating functions with estimate plugged in do not sum to within required tolerance of zero: {estimating_function_sum}"
        )

    return running_meat_matrix / len(user_ids)


# TODO: Handle get_loss_gradient generic interface.  Probably need some function that just takes a
# study df, state feats, treat feats, and theta est
# TODO: Also think about how to specify whether to treat something as action probabilities
# TODO: idk if beta dim should be included in here
# TODO: doc string
# TODO: Why am I passing in update times again? Can I just derive from study df?
# TODO: Do the three checks in the existing after study file
# TODO: This is a hotspot for update time logic to be removed
def form_bread_inverse_matrix(
    upper_left_bread_inverse,
    max_t,
    algo_stats_dict,
    update_times,
    beta_dim,
    theta_dim,
    user_ids,
    loss_gradients,
    loss_hessians,
    loss_gradient_derivatives_wrt_pi,
):
    existing_rows = upper_left_bread_inverse.shape[0]

    # This is useful for sweeping through the decision times between updates
    # but critically also those after the final update
    update_times_and_upper_limit = (
        update_times if update_times[-1] == max_t + 1 else update_times + [max_t + 1]
    )

    # Begin by creating a few convenience data structures for the mixed theta/beta derivatives
    # that are most easily created for many decision times at once, whereas the following loop is
    # over update times.  We will pull appropriate quantities from here during iterations of the
    # loop.

    # This computes derivatives of the theta estimating function wrt the action probabilities
    # vector, which importantly has an element for *every* decision time.  We will later do the
    # work to multiply these by derivatives of pi with respect to beta, thus getting the quantities
    # we really want via the chain rule, and also summing terms that correspond to the *same* betas
    # behind the scenes.
    # NOTE THAT COLUMN INDEX i CORRESPONDS TO DECISION TIME i+1!
    # TODO: This squeeze is a little sketchy... It might be nice to squeeze at the time they are
    # computed. Note there is also a corresponding RL squeeze, but it happens closer to
    # the gradient computation.
    mixed_theta_pi_loss_derivatives_by_user_id = {
        user_id: loss_gradient_derivatives_wrt_pi[i].squeeze()
        for i, user_id in enumerate(user_ids)
    }

    # This simply collects the pi derivatives with respect to betas for all
    # decision times for each user, reorganizing existing data from the RL side.
    # The one complication is that we add some padding of zeros for decision
    # times before the first update to be in correspondence with the above data
    # structure.
    # See the analogous comment in the construction of the RL portion of the
    # matrix for more details on why we limit to t >= update_times[0]. In short
    # we do not need the values before that, but want to allow them to be given.
    # Whether they are given or not, we choose to put zeros in their place.
    # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
    pi_derivatives_by_user_id = {
        user_id: jnp.pad(
            jnp.array(
                [
                    t_stats_dict["pi_gradients_by_user_id"][user_id]
                    for t, t_stats_dict in algo_stats_dict.items()
                    if t >= update_times[0]
                ]
            ),
            pad_width=((update_times[0] - 1, 0), (0, 0)),
        )
        for user_id in user_ids
    }

    # Think of each iteration of this loop as creating one off-diagonal term in
    # the final (block) row
    bottom_left_row_blocks = []
    for i in range(len(update_times)):
        lower_t = update_times_and_upper_limit[i]
        upper_t = update_times_and_upper_limit[i + 1]
        running_entry_holder = jnp.zeros((theta_dim, beta_dim))

        # This loop calculates the per-user quantities that will be
        # averaged for the final matrix entries
        for j, user_id in enumerate(user_ids):
            # 1. We first form the outer product of the estimating equation for theta
            # and the sum of the weight gradients with respect to beta for the
            # corresponding decision times

            theta_loss_gradient = loss_gradients[j]

            weight_gradient_sum = jnp.zeros(beta_dim)

            # This loop iterates over decision times in slices between updates
            # to collect the right weight gradients
            # Note these may look more sparse than expected due to clipping, which
            # produces zero gradients when limits are hit.
            for t in range(lower_t, upper_t):
                weight_gradient_sum += algo_stats_dict[t][
                    "weight_gradients_by_user_id"
                ][user_id]
            running_entry_holder += jnp.outer(theta_loss_gradient, weight_gradient_sum)

            # 2. We now calculate mixed derivatives of the loss wrt theta and then beta. This piece
            # is a bit intricate; we only have the theta loss function in terms of the pis,
            # and the *values* of the pi derivatives wrt to betas available here, since the actual
            # pi functions are the domain of the RL side. The loss function also gets an action
            # probability for all decision times, not knowing which correspond to which
            # betas behind the scenes, so our tasks are to
            # 1. multiply these theta derivatives wrt pi for each relevant decision
            #    time by the corresponding pi derivative wrt beta
            # 2. sum together the products from the previous step that actually
            #    correspond to the same betas
            # The loop we are currently in is doing this for just the bucket of decision
            # times currently under consideration.

            # Multiply just the appropriate segments of the precomputed
            # mixed theta pi loss derivative matrix for the given user and
            # the precollected pi beta derivative matrix for the user. These
            # segments are simply those that correspond to all the decision times
            # in the current slice between updates under consideration.
            # NOTE THAT OUR HELPER DATA STRUCTURES ARE 0-INDEXED, SO WE SUBTRACT
            # 1 FROM OUR TIME BOUNDS.
            # NOTE we could also do something like a join on policy number,
            # then multiply and sum in groups--may be simpler to think about
            # than dealing with spans of update times
            mixed_theta_beta_loss_derivative = jnp.matmul(
                mixed_theta_pi_loss_derivatives_by_user_id[user_id][
                    :,
                    lower_t - 1 : upper_t - 1,
                ],
                pi_derivatives_by_user_id[user_id][
                    lower_t - 1 : upper_t - 1,
                    :,
                ],
            )

            running_entry_holder += mixed_theta_beta_loss_derivative

        bottom_left_row_blocks.append(running_entry_holder / len(user_ids))
    bottom_right_hessian = jnp.mean(loss_hessians, axis=0)
    return jnp.block(
        [
            [
                upper_left_bread_inverse,
                jnp.zeros((existing_rows, theta_dim)),
            ],
            [
                jnp.block(bottom_left_row_blocks),
                bottom_right_hessian,
            ],
        ]
    )


# TODO: Needs tests
# TODO: Complete docstring
def get_classical_sandwich_var(theta_dim, loss_gradients, loss_hessians):
    """
    Forms standard sandwich variance estimator for inference (thetahat)

    Input:

    Output:
    - Sandwich variance estimator matrix (size dim_theta by dim_theta)
    """

    logger.info("Forming classical sandwich variance estimator.")
    num_users = len(loss_gradients)

    logger.info("Forming classical meat.")
    running_meat_matrix = np.zeros((theta_dim, theta_dim)).astype(jnp.float32)
    for loss_gradient in loss_gradients:
        user_meat_vector = loss_gradient.reshape(-1, 1)
        running_meat_matrix += np.outer(user_meat_vector, user_meat_vector)

    meat = running_meat_matrix / num_users

    logger.info("Forming classical bread inverse.")
    normalized_hessian = np.mean(loss_hessians, axis=0)
    logger.info("Classical bread (pre-inversion):")
    logger.info(normalized_hessian)
    logger.info("Classical meat:")
    logger.info(meat)

    # degrees of freedom adjustment
    # TODO: Reinstate? Provide reference? Mentioned in sandwich package
    # This is HC1 correction
    # Should we use something other than theta_dim for d?
    # meat = meat * (num_users - 1) / (num_users - theta_dim)

    logger.info("Inverting classical bread and combining ingredients.")
    inv_hessian = invert_matrix_and_check_conditioning(normalized_hessian)
    sandwich_var = (inv_hessian @ meat @ inv_hessian.T) / num_users

    logger.info("Finished forming classical sandwich variance estimator.")

    return sandwich_var


if __name__ == "__main__":
    cli()
