import pickle
import os
import logging
import cProfile
from pstats import Stats
import warnings
import pathlib
import glob

import click
import jax
import numpy as np
from jax import numpy as jnp
from sklearn.linear_model import LinearRegression
import scipy


from helper_functions import invert_matrix_and_check_conditioning

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


# TODO: Think about interface here.  User should probably specify model, we create loss from it
def get_loss(
    theta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    action1probs,
    action_centering,
):
    theta_0 = theta_est[: base_states.shape[1]].reshape(-1, 1)
    theta_1 = theta_est[base_states.shape[1] :].reshape(-1, 1)

    actions = jnp.where(
        action_centering, actions.astype(jnp.float32) - action1probs, actions
    )

    return jnp.sum(
        (
            rewards
            - jnp.matmul(base_states, theta_0)
            - jnp.matmul(actions * treat_states, theta_1)
        )
        ** 2
    )


# For the loss gradients, we can form the sum of all users values and differentiate that with one
# call. Instead, this alternative structure which generalizes to the pi function case.
def get_loss_gradients_batched(
    theta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
    action1probs_batch,
    action_centering,
):
    return jax.vmap(
        fun=jax.grad(get_loss),
        in_axes=(None, 2, 2, 2, 2, 2, None),
        out_axes=0,
    )(
        theta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
        action1probs_batch,
        action_centering,
    )


def get_loss_hessians_batched(
    theta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
    action1probs_batch,
    action_centering,
):
    return jax.vmap(
        fun=jax.hessian(get_loss),
        in_axes=(None, 2, 2, 2, 2, 2, None),
        out_axes=0,
    )(
        theta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
        action1probs_batch,
        action_centering,
    )


def get_loss_gradient_derivatives_wrt_pi_batched(
    theta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
    action1probs_batch,
    action_centering,
):
    return jax.vmap(
        fun=jax.jacrev(jax.grad(get_loss), 5),
        in_axes=(None, 2, 2, 2, 2, 2, None),
        out_axes=0,
    )(
        theta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
        action1probs_batch,
        action_centering,
    )


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
        if i % (len(filenames) // 10) == 0:
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
    empirical_var_normalized = np.cov(theta_estimates.T, ddof=0)
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


# TODO: Eventually these commands will just take like the algo statistics object and
# anything else we need after parsing by intermediate package.
# TODO: This command should handle saving the pickled objects eventually if it
# sticks around.
@cli.command()
@click.option(
    "--input_folder",
    type=click.Path(exists=True),
    help="A parent folder containing a series of subfolders with the required files",
    required=True,
)
@click.option(
    "--study_dataframe_pickle_filename",
    help="The name of the pickled pandas dataframe in each folder",
    required=True,
)
@click.option(
    "--rl_algorithm_object_pickle_filename",
    help="The name of the pickled RL algorithm in each folder",
    required=True,
)
@click.option(
    "--profile",
    default=False,
    is_flag=True,
    help="If supplied, the important computations will be profiled with summary output shown",
)
@click.option(
    "--action_centering",  # TODO: This needs to be handled more generall and not as an int
    type=int,
    default=0,
)
def analyze_multiple_datasets_and_compare_to_empirical_variance(
    input_folder,
    study_dataframe_pickle_filename,
    rl_algorithm_object_pickle_filename,
    profile,
    action_centering,
):
    """
    For each of the supplied input_folders, extract the pickled study dataframe
    and RL algorithm object using the supplied filenames. Analyze each
    dataset and compute the empirical variance to compare with the mean adaptive
    sandwich variance.
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    theta_estimates = []
    adaptive_sandwich_var_estimates = []
    classical_sandwich_var_estimates = []

    for subfolder in os.listdir(input_folder):
        # We care about folders, not files
        if os.path.isfile(os.path.join(input_folder, subfolder)):
            continue

        logger.info("Processing folder %s", subfolder)
        # Check to make sure each subfolder contains the two required files
        contains_study_df = False
        contains_rl_alg = False

        for filename in os.listdir(os.path.join(input_folder, subfolder)):
            # Skip other files
            if filename not in {
                study_dataframe_pickle_filename,
                rl_algorithm_object_pickle_filename,
            }:
                continue

            # Check that objects at these paths are actually files (not folders)
            f = os.path.join(input_folder, subfolder, filename)
            if not os.path.isfile(f):
                raise RuntimeError(f"Required path {f} exists but it is not a file")

            # Record when we've found files of each type
            if filename == study_dataframe_pickle_filename:
                contains_study_df = True
            elif filename == rl_algorithm_object_pickle_filename:
                contains_rl_alg = True

            if contains_study_df and contains_rl_alg:
                break
        else:
            raise RuntimeError(
                f"Folder {subfolder} did not contain at least one the required files {study_dataframe_pickle_filename} and {rl_algorithm_object_pickle_filename}"
            )

        logger.info("Contains required files, let's proceed to analysis.")
        # If we got here, both required files were found. Analyze this dataset.
        with open(
            os.path.join(input_folder, subfolder, study_dataframe_pickle_filename),
            "+rb",
        ) as study_dataframe_pickle:
            with open(
                os.path.join(
                    input_folder, subfolder, rl_algorithm_object_pickle_filename
                ),
                "+rb",
            ) as rl_algorithm_object_pickle:
                study_df = pickle.load(study_dataframe_pickle)
                study_RLalg = pickle.load(rl_algorithm_object_pickle)

                (
                    theta_est,
                    adaptive_sandwich_var,
                    classical_sandwich_var,
                ) = analyze_dataset_inner(study_df, study_RLalg, bool(action_centering))

                theta_estimates.append(theta_est)
                adaptive_sandwich_var_estimates.append(adaptive_sandwich_var)
                classical_sandwich_var_estimates.append(classical_sandwich_var)

    theta_estimates = np.array(theta_estimates)
    adaptive_sandwich_var_estimates = np.array(adaptive_sandwich_var_estimates)
    classical_sandwich_var_estimates = np.array(classical_sandwich_var_estimates)

    theta_estimate = np.mean(theta_estimates, axis=0)
    empirical_var_normalized = np.cov(theta_estimates.T, ddof=0)
    mean_adaptive_sandwich_var_estimate = np.mean(
        adaptive_sandwich_var_estimates, axis=0
    )
    mean_classical_sandwich_var_estimate = np.mean(
        classical_sandwich_var_estimates, axis=0
    )

    print(f"\nParameter estimate:\n{theta_estimate}")
    print(f"\nEmpirical variance:\n{empirical_var_normalized}")
    print(
        f"\nAdaptive sandwich variance estimate:\n{mean_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\nClassical sandwich variance estimate:\n{mean_classical_sandwich_var_estimate}\n",
    )

    # TODO: Save results to sensible output directory, perhaps where input data is.

    if profile:
        pr.disable()
        stats = Stats(pr)
        stats.sort_stats("cumtime").print_stats(50)


# TODO: ADD redo analysis toggle?
@cli.command()
@click.option(
    "--study_dataframe_pickle",
    type=click.File("rb"),
    help="Pickled pandas dataframe in correct format (see contract/readme)",
    required=True,
)
@click.option(
    "--rl_algorithm_object_pickle",
    type=click.File("rb"),
    help="Pickled RL algorithm object in correct format (see contract/readme)",
    required=True,
)
@click.option(
    "--profile",
    default=False,
    is_flag=True,
    help="If supplied, the important computations will be profiled with summary output shown",
)
@click.option(
    "--action_centering",  # TODO: This needs to be handled more generall and not as an int
    type=int,
    default=0,
)
def analyze_dataset(
    study_dataframe_pickle, rl_algorithm_object_pickle, profile, action_centering
):

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # Load study data
    study_df = pickle.load(study_dataframe_pickle)
    study_RLalg = pickle.load(rl_algorithm_object_pickle)

    # Analyze data
    theta_est, adaptive_sandwich_var_estimate, classical_sandwich_var_estimate = (
        analyze_dataset_inner(study_df, study_RLalg, bool(action_centering))
    )

    # Write analysis results to same directory as input files
    folder_path = pathlib.Path(study_dataframe_pickle.name).parent.resolve()
    with open(f"{folder_path}/analysis.pkl", "wb") as f:
        pickle.dump(
            {
                "theta_est": theta_est,
                "adaptive_sandwich_var_estimate": adaptive_sandwich_var_estimate,
                "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
            },
            f,
        )

    print(f"\nParameter estimate:\n {theta_est}")
    print(f"\nAdaptive sandwich variance estimate:\n {adaptive_sandwich_var_estimate}")
    print(
        f"\nClassical sandwich variance estimate:\n {classical_sandwich_var_estimate}\n"
    )

    if profile:
        pr.disable()
        stats = Stats(pr)
        stats.sort_stats("cumtime").print_stats(50)


# TODO: docstring
# TODO: Collect all jax things and then einsum away?
def analyze_dataset_inner(study_df, study_RLalg, action_centering):

    # List of times that were the first applicable time for some update
    # Sorting shouldn't be necessary, as insertion order should be chronological
    # but we do it just in case.
    update_times = sorted(
        [
            t
            for t, value in study_RLalg.algorithm_statistics_by_calendar_t.items()
            if "loss_gradients_by_user_id" in value
        ]
    )

    breakpoint()
    # TODO: state features should not be the same as RL alg for full generality
    # Estimate the inferential target using the supplied study data.
    logger.info("Forming theta estimate.")
    theta_est = estimate_theta(
        study_df, study_RLalg.state_feats, study_RLalg.treat_feats, action_centering
    )

    logger.info("Forming adaptive sandwich variance estimator.")

    logger.info("Calculating all derivatives needed with JAX")
    user_ids, loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
        collect_derivatives(
            study_df,
            study_RLalg.state_feats,
            study_RLalg.treat_feats,
            theta_est,
            action_centering,
        )
    )

    # TODO: state features should not be the same as RL alg for full generality
    logger.info("Forming adaptive bread inverse and inverting.")
    max_t = study_df.calendar_t.max()
    theta_dim = len(theta_est)
    joint_bread_inverse_matrix = form_bread_inverse_matrix(
        study_RLalg.upper_left_bread_inverse,
        max_t,
        study_RLalg.algorithm_statistics_by_calendar_t,
        update_times,
        study_RLalg.beta_dim,
        theta_dim,
        user_ids,
        loss_gradients,
        loss_hessians,
        loss_gradient_pi_derivatives,
    )
    joint_bread_matrix = invert_matrix_and_check_conditioning(
        joint_bread_inverse_matrix
    )

    logger.info("Forming adaptive meat.")
    # TODO: Should I do DOF adjustment as for classical? Probably doesn't make sense to do in
    # one case but not the other.
    joint_meat_matrix = form_meat_matrix(
        theta_dim,
        update_times,
        study_RLalg.beta_dim,
        study_RLalg.algorithm_statistics_by_calendar_t,
        user_ids,
        loss_gradients,
    )

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
        theta_est,
        joint_adaptive_variance[-len(theta_est) :, -len(theta_est) :],
        get_classical_sandwich_var(theta_dim, loss_gradients, loss_hessians),
    )


# TODO: Think about user interface.  Do they give whole estimate theta function? or simply a model
# spec and we do the fitting within some framework
# TODO: Should we specify the format of study df or allow flexibility?
# TODO: doc string
def estimate_theta(study_df, state_feats, treat_feats, action_centering):
    # Note that the intercept is included in the features already (col of 1s)
    linear_model = LinearRegression(fit_intercept=False)

    # Note the role of the action centering flag in here in determining whether
    # we subtract action probabilities from actions (multiplying by a boolean
    # in python is like multiplying by 1 if True and 0 if False).
    trimmed_df = study_df[state_feats].copy()
    for feat in treat_feats:
        trimmed_df[f"action:{feat}"] = study_df[feat] * (
            study_df["action"] - (study_df["action1prob"] * action_centering)
        )

    linear_model.fit(trimmed_df, study_df["reward"])

    return linear_model.coef_


# TODO: Just use dicts keyed on user id...
def collect_derivatives(
    study_df, state_feats, treat_feats, theta_est, action_centering
):
    batched_base_states_list = []
    batched_treat_states_list = []
    batched_actions_list = []
    batched_rewards_list = []
    batched_action1probs_list = []

    user_ids = study_df.user_id.unique()
    for user_id in user_ids:
        filtered_user_data = study_df.loc[study_df.user_id == user_id]
        batched_base_states_list.append(
            get_base_states(filtered_user_data, state_feats)
        )
        batched_treat_states_list.append(
            get_treat_states(filtered_user_data, treat_feats)
        )
        batched_actions_list.append(get_actions(filtered_user_data))
        batched_rewards_list.append(get_rewards(filtered_user_data))
        batched_action1probs_list.append(get_action1probs(filtered_user_data))

    # TODO: dstack breaks with incremental recruitment
    batched_base_states_tensor = jnp.dstack(batched_base_states_list)
    batched_treat_states_tensor = jnp.dstack(batched_treat_states_list)
    batched_actions_tensor = jnp.dstack(batched_actions_list)
    batched_rewards_tensor = jnp.dstack(batched_rewards_list)
    batched_action1probs_tensor = jnp.dstack(batched_action1probs_list)

    loss_gradients = get_loss_gradients_batched(
        theta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        batched_actions_tensor,
        batched_rewards_tensor,
        batched_action1probs_tensor,
        action_centering,
    )

    loss_hessians = get_loss_hessians_batched(
        theta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        batched_actions_tensor,
        batched_rewards_tensor,
        batched_action1probs_tensor,
        action_centering,
    )

    loss_gradient_pi_derivatives = get_loss_gradient_derivatives_wrt_pi_batched(
        theta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        batched_actions_tensor,
        batched_rewards_tensor,
        batched_action1probs_tensor,
        action_centering,
    )

    return user_ids, loss_gradients, loss_hessians, loss_gradient_pi_derivatives


# TODO: doc string
# TODO: rewrite as einsum?
def form_meat_matrix(
    theta_dim, update_times, beta_dim, algo_stats_dict, user_ids, loss_gradients
):
    num_rows_cols = beta_dim * len(update_times) + theta_dim
    running_meat_matrix = jnp.zeros((num_rows_cols, num_rows_cols)).astype(jnp.float32)
    estimating_function_sum = jnp.zeros((num_rows_cols, 1))

    for i, user_id in enumerate(user_ids):
        user_meat_vector = jnp.concatenate(
            [
                algo_stats_dict[t]["loss_gradients_by_user_id"][user_id]
                for t in update_times
            ]
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


# TODO: Docstring
def get_base_states(df, state_feats):
    base_states = df[state_feats].to_numpy()
    return jnp.array(base_states)


def get_treat_states(df, treat_feats):
    treat_states = df[treat_feats].to_numpy()
    return jnp.array(treat_states)


# TODO: Allow general reward column names
def get_rewards(df):
    rewards = df["reward"].to_numpy().reshape(-1, 1)
    return jnp.array(rewards)


# TODO: Allow general action column names
def get_actions(df):
    actions = df["action"].to_numpy().reshape(-1, 1)
    return jnp.array(actions)


# TODO: Allow general action column names
def get_action1probs(df):
    action1probs = df["action1prob"].to_numpy().reshape(-1, 1)
    return jnp.array(action1probs)


# TODO: Doc string
def get_user_states(study_df, state_feats, treat_feats, user_id):
    """
    Extract just the states for the given user in the given study_df as a
    tuple of numpy (column) vectors.
    """
    user_df = study_df.loc[study_df.user_id == user_id]
    base_states = user_df[state_feats].to_numpy()
    treat_states = user_df[treat_feats].to_numpy()
    return (base_states, treat_states)


# TODO: Handle get_loss_gradient generic interface.  Probably need some function that just takes a
# study df, state feats, treat feats, and theta est
# TODO: Also think about how to specify whether to treat something as action probabilities
# TODO: idk if beta dim should be included in here
# TODO: doc string
# TODO: Why am I passing in update times again? Can I just derive from study df?
# TODO: Do the three checks in the existing after study file
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
    # TODO: Make sure this makes sense for decisions between updates > 1
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
    # NOTE that JAX treats positional args as keyword args if they are *supplied* with name=val
    # syntax.  So though supplying these arg names is a good practice for readability, it has
    # unexpected consequences in this case. Just noting this because it was tricky to debug here.
    mixed_theta_pi_loss_derivatives_by_user_id = {
        user_id: loss_gradient_derivatives_wrt_pi[i].squeeze()
        for i, user_id in enumerate(user_ids)
    }
    # TODO: Handle missing data?
    # This simply collects the pi derivatives with respect to betas for all
    # decision times for each user, reorganizing existing data from the RL side.
    # The one complication is that we add some padding of zeros for decision
    # times before the first update to be in correspondence with the above data
    # structure.
    # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
    pi_derivatives_by_user_id = {
        user_id: jnp.pad(
            jnp.array(
                [
                    t_stats_dict["pi_gradients_by_user_id"][user_id]
                    for t_stats_dict in algo_stats_dict.values()
                ]
            ),
            pad_width=((update_times[0] - 1, 0), (0, 0)),
        )
        for user_id in user_ids
    }

    # Think of each iteration of this loop as creating one term in the final (block) row
    bottom_left_row_blocks = []
    for i in range(len(update_times)):
        lower_t = update_times_and_upper_limit[i]
        upper_t = update_times_and_upper_limit[i + 1]
        running_entry_holder = jnp.zeros((theta_dim, theta_dim))

        # This loop calculates the per-user quantities that will be
        # averaged for the final matrix entries
        for i, user_id in enumerate(user_ids):
            # 1. We first form the outer product of the estimating equation for theta
            # and the sum of the weight gradients with respect to beta for the
            # corresponding decision times

            theta_loss_gradient = loss_gradients[i]

            weight_gradient_sum = jnp.zeros(beta_dim)

            # This loop iterates over decision times in slices between updates
            # to collect the right weight gradients
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
            # TODO: *Only* do this when action centering or otherwise using pis
            # NOTE THAT OUR HELPER DATA STRUCTURES ARE 0-INDEXED, SO WE SUBTRACT
            # 1 FROM OUR TIME BOUNDS.
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
# TODO: Don't recompute things already computed for adaptive sandwich (or vice versa)
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

    # degrees of freedom adjustment
    # TODO: Provide reference
    meat = meat * (num_users - 1) / (num_users - theta_dim)

    logger.info("Inverting classical bread and combining ingredients.")
    inv_hessian = invert_matrix_and_check_conditioning(normalized_hessian)
    sandwich_var = (inv_hessian @ meat @ inv_hessian.T) / num_users

    logger.info("Finished forming classical sandwich variance estimator.")

    return sandwich_var


if __name__ == "__main__":
    cli()
