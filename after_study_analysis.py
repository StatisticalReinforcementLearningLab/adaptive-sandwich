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

import calculate_RL_derivatives


from helper_functions import invert_matrix_and_check_conditioning

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# TODO: Break this file up


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
# TODO: Handle raw timestamps instead of calendar time index? For now I'm requiring it
# TODO: Make sure user id column name is actually respected. There are .user_id's lingering
@cli.command()
@click.option(
    "--study_df_pickle",
    type=click.File("rb"),
    help="Pickled pandas dataframe in correct format (see contract/readme)",
    required=True,
)
@click.option(
    "--beta_df_pickle",
    type=click.File("rb"),
    help="Pickled pandas dataframe in correct format (see contract/readme)",
    required=True,
)
@click.option(
    "--action_prob_func_filename",
    type=click.File("rb"),
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
    "--RL_loss_func_filename",
    type=click.Path(exists=True),
    help="File that contains the per-user loss function used to determine the RL parameters at each update and relevant imports.  The filename will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--RL_loss_func_args_pickle",
    type=click.File("rb"),
    help="Pickled dictionary that contains the RL loss function arguments for all update times for all users",
    required=True,
)
@click.option(
    "--RL_loss_func_args_beta_index",
    type=int,
    required=True,
    help="Index of the RL parameter vector beta in the tuple of RL loss func args",
)
@click.option(
    "--RL_loss_func_args_action_prob_index",
    type=int,
    default=-1,
    help="Index of the action probability in the tuple of RL loss func args, if applicable",
)
@click.option(
    "--covariate_names",
    type=str,
    help="If supplied, the important computations will be profiled with summary output shown",
    required=True,
)
@click.option(
    "--profile",
    default=False,
    is_flag=True,
    help="If supplied, the important computations will be profiled with summary output shown",
)
@click.option(
    "--action_centering",  # TODO: This needs to be handled more generally and not as an int
    type=int,
    default=0,
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
def analyze_dataset(
    study_df_pickle,
    beta_df_pickle,
    action_prob_func_filename,
    action_prob_func_args_pickle,
    action_prob_func_args_beta_index,
    RL_loss_func_filename,
    RL_loss_func_args_pickle,
    RL_loss_func_args_beta_index,
    RL_loss_func_args_action_prob_index,
    covariate_names_str,
    profile,
    action_centering,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
):
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    if action_centering:
        logging.info("Action centering is ENABLED for inference.")

    # Load study data
    study_df = pickle.load(study_df_pickle)

    # TODO: Do I even need this beta df?
    beta_df = pickle.load(beta_df_pickle)
    beta_dim = beta_df.shape[1]

    action_prob_func_args = pickle.load(action_prob_func_args_pickle)
    RL_loss_func_args = pickle.load(RL_loss_func_args_pickle)

    # TODO: Data integrity checks.
    # Reconstruct action probabilites now or later?

    # I check estimating function sum 0 later, but differentiate between
    # RL and beta side? Also could move check here. Might be  nice to have all
    # checks in same place.

    # Make sure study df sorted within users and across users? .sort_values(by=["user_id", "calendar_t"])

    # Make sure every user has entry for union of all decision times across
    # all users.

    # Make sure in study is never on for more than one stretch

    # Make sure right number of in study rows per user?? May not be precise

    # Possibly make sure no real-looking data when in study is off

    # Make sure function args for all users for all t even if not in study
    # policy num the same for all users for each calendar t.  Something that adds
    # nothing to gradients when users are not in the study is needed... maybe
    # think of way to make this happen more directly in response to in study
    # indicator. Well, can pass in zeros or NAs but not use computed gradients,
    # automatically spit out gradient zero somehow.

    # I think I'm agnostic to indexing of calendar times but should check because
    # otherwise need to add a check here to verify required format.

    # Verify actions binary

    # Verify action column present in study df

    # Verify policy num column present in both beta df and study df, and joinable,
    # for instance no policy nums in study df not present in beta df.  Other way
    # is acceptable maybe but could be a warning.

    # Verify in study column present in study df

    # If no action probabilites vector is specified, ask for verification that they are not used in loss/estimating function(s)

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
        RL_loss_func_filename,
        RL_loss_func_args,
        RL_loss_func_args_beta_index,
        RL_loss_func_args_action_prob_index,
    )
    upper_left_bread_inverse = calculate_upper_left_bread_inverse(
        study_df, beta_dim, algorithm_statistics_by_calendar_t
    )

    covariate_names = covariate_names_str.split(",")

    # Analyze data
    theta_est, adaptive_sandwich_var_estimate, classical_sandwich_var_estimate = (
        analyze_dataset_inner(
            study_df,
            beta_dim,
            algorithm_statistics_by_calendar_t,
            upper_left_bread_inverse,
            bool(action_centering),
            covariate_names,
            in_study_col_name,
        )
    )

    # Write analysis results to same directory as input files
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
    RL_loss_func_filename,
    RL_loss_func_args,
    RL_loss_func_args_beta_index,
    RL_loss_func_args_action_prob_index,
):
    pi_and_weight_gradients_by_calendar_t = (
        calculate_RL_derivatives.calculate_pi_and_weight_gradients(
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

    RL_update_derivatives_by_calendar_t = (
        calculate_RL_derivatives.calculate_loss_derivatives(
            study_df,
            RL_loss_func_filename,
            RL_loss_func_args,
            RL_loss_func_args_beta_index,
            RL_loss_func_args_action_prob_index,
            policy_num_col_name,
            calendar_t_col_name,
        )
    )

    merged_dict = pi_and_weight_gradients_by_calendar_t
    for t, t_dict in pi_and_weight_gradients_by_calendar_t.items():
        merged_dict[t] = {
            **t_dict,
            **RL_update_derivatives_by_calendar_t.get(t, {}),
        }

    return merged_dict


# TODO: docstring
def calculate_upper_left_bread_inverse(
    study_df, beta_dim, algorithm_statistics_by_calendar_t
):

    # Form the dimensions for our bread matrix portion (pre-inverting)
    num_updates = None
    overall_dim = beta_dim * num_updates
    output_matrix = jnp.zeros((overall_dim, overall_dim))

    # List of times that were the first applicable time for some update
    # TODO: sort to not rely on insertion order?
    # TODO: use policy_num in df? alg statistics potentially ok too though.
    next_times_after_update = [
        t
        for t, value in algorithm_statistics_by_calendar_t.items()
        if "loss_gradients_by_user_id" in value
    ]

    user_ids = study_df.user_id.unique()
    num_users = len(user_ids)

    # This simply collects the pi derivatives with respect to betas for all
    # decision times for each user. The one complication is that we add some
    # padding of zeros for decision times before the first update to make
    # indexing simpler below.
    # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
    pi_derivatives_by_user_id = {
        user_id: jnp.pad(
            jnp.array(
                [
                    t_dict["pi_gradients_by_user_id"][user_id]
                    for t_dict in algorithm_statistics_by_calendar_t.values()
                ]
            ),
            pad_width=((next_times_after_update[0] - 1, 0), (0, 0)),
        )
        for user_id in user_ids
    }

    # This loop iterates over all times that were the first applicable time
    # for a non-initial policy. Take care to note that update_idx starts at 0.
    # Think of each iteration of this loop as creating a (block) row of the matrix
    for update_idx, update_t in enumerate(next_times_after_update):
        logger.info("Processing update time %s.", update_t)
        t_stats_dict = algorithm_statistics_by_calendar_t[update_t]

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


# TODO: docstring
def analyze_dataset_inner(
    study_df,
    beta_dim,
    algorithm_statistics_by_calendar_t,
    upper_left_bread_inverse,
    action_centering,
    covariate_names,
    in_study_column,
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

    # TODO: state features should not be the same as RL alg for full generality
    # Estimate the inferential target using the supplied study data.
    # TODO: We may not want to estimate theta in general... too complicated.
    logger.info("Forming theta estimate.")

    theta_est = estimate_theta(
        study_df,
        covariate_names,
        action_centering,
        in_study_column,
    )

    logger.info("Forming adaptive sandwich variance estimator.")

    logger.info("Calculating all derivatives needed with JAX")
    user_ids, loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
        collect_derivatives(
            study_df,
            covariate_names,
            theta_est,
            action_centering,
        )
    )

    # TODO: state features should not be the same as RL alg for full generality
    logger.info("Forming adaptive bread inverse and inverting.")
    max_t = study_df.calendar_t.max()
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
        theta_est,
        joint_adaptive_variance[-len(theta_est) :, -len(theta_est) :],
        get_classical_sandwich_var(theta_dim, loss_gradients, loss_hessians),
    )


# TODO: Think about user interface.  Do they give whole estimate theta function? or simply a model
# spec and we do the fitting within some framework
# TODO: Should we specify the format of study df or allow flexibility?
# TODO: doc string
def estimate_theta(study_df, covariate_names, action_centering, in_study_column):
    # Note that the intercept is included in the features already (col of 1s)
    # in the way we typically run this
    linear_model = LinearRegression(fit_intercept=False)

    # Note the role of the action centering flag in here in determining whether
    # we subtract action probabilities from actions (multiplying by a boolean
    # in python is like multiplying by 1 if True and 0 if False).
    in_study_bool = study_df[in_study_column] == 1
    trimmed_df = study_df.loc[in_study_bool, covariate_names].copy()
    in_study_df = study_df[in_study_bool]
    for feat in covariate_names:
        trimmed_df[f"action:{feat}"] = in_study_df[feat] * (
            in_study_df["action"] - (in_study_df["action1prob"] * action_centering)
        )

    linear_model.fit(trimmed_df, in_study_df["reward"])

    return linear_model.coef_


# TODO: Just use dicts keyed on user id...
def collect_derivatives(study_df, covariate_names, theta_est, action_centering):
    batched_base_states_list = []
    batched_treat_states_list = []
    batched_actions_list = []
    batched_rewards_list = []
    batched_action1probs_list = []

    user_ids = study_df.user_id.unique()
    for user_id in user_ids:
        filtered_user_data = study_df.loc[study_df.user_id == user_id]
        batched_base_states_list.append(
            get_base_states(filtered_user_data, covariate_names)
        )
        batched_treat_states_list.append(
            get_treat_states(filtered_user_data, covariate_names)
        )
        batched_actions_list.append(get_actions(filtered_user_data))
        batched_rewards_list.append(get_rewards(filtered_user_data))
        batched_action1probs_list.append(get_action1probs(filtered_user_data))

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


# TODO: Docstring
def get_base_states(df, state_feats, in_study_col="in_study"):
    df.loc[df[in_study_col] == 0, state_feats] = 0
    base_states = df[state_feats].to_numpy()
    return jnp.array(base_states)


# TODO: We can really only have one state-getting function in general
def get_treat_states(df, treat_feats, in_study_col="in_study"):
    df.loc[df[in_study_col] == 0, treat_feats] = 0
    treat_states = df[treat_feats].to_numpy()
    return jnp.array(treat_states)


# TODO: Type conversion here a little sloppy
def get_rewards(df, in_study_col="in_study", reward_col="reward"):
    df.loc[df[in_study_col] == 0, reward_col] = 0
    rewards = df[reward_col].to_numpy(dtype="float64").reshape(-1, 1)
    return jnp.array(rewards)


# TODO: Type conversion here a little sloppy
def get_actions(df, in_study_col="in_study", action_col="action"):
    df.loc[df[in_study_col] == 0, action_col] = 0
    actions = df[action_col].to_numpy(dtype="int32").reshape(-1, 1)
    return jnp.array(actions)


# TODO: Type conversion here a little sloppy
def get_action1probs(df, in_study_col="in_study", actionprob_col="action1prob"):
    df.loc[df[in_study_col] == 0, actionprob_col] = 0
    action1probs = df[actionprob_col].to_numpy(dtype="float64").reshape(-1, 1)
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
    mixed_theta_pi_loss_derivatives_by_user_id = {
        user_id: loss_gradient_derivatives_wrt_pi[i].squeeze()
        for i, user_id in enumerate(user_ids)
    }

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

    # Think of each iteration of this loop as creating one off-diagonal term in
    # the final (block) row
    bottom_left_row_blocks = []
    for i in range(len(update_times)):
        lower_t = update_times_and_upper_limit[i]
        upper_t = update_times_and_upper_limit[i + 1]
        running_entry_holder = jnp.zeros((theta_dim, theta_dim))

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
# TODO: Don't recompute things already computed for adaptive sandwich (or vice versa)
# TODO: verify works with incremental recruitment
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
    # meat = meat * (num_users - 1) / (num_users - theta_dim)

    logger.info("Inverting classical bread and combining ingredients.")
    inv_hessian = invert_matrix_and_check_conditioning(normalized_hessian)
    sandwich_var = (inv_hessian @ meat @ inv_hessian.T) / num_users

    logger.info("Finished forming classical sandwich variance estimator.")

    return sandwich_var


if __name__ == "__main__":
    cli()
