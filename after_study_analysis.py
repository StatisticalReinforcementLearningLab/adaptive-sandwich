import os
import pickle

import click
import jax
import numpy as np
from jax import numpy as jnp
from sklearn.linear_model import LinearRegression

from helper_functions import get_user_action1probs, get_user_actions, get_user_rewards


# TODO: Think about interface here.  User should probably specify model, we create loss from it
def get_loss(theta_est, base_states, treat_states, actions, rewards, action1probs=None):
    theta_0 = theta_est[: base_states.shape[1]].reshape(-1, 1)
    theta_1 = theta_est[base_states.shape[1] :].reshape(-1, 1)

    # Perform action centering if given action probabilities
    if action1probs is not None:
        # TODO: deal with types more cleanly?
        actions = actions.astype(jnp.float64)
        actions -= action1probs

    return jnp.sum(
        (
            rewards
            - jnp.matmul(base_states, theta_0)
            - jnp.matmul(actions * treat_states, theta_1)
        )
        ** 2
    )


get_loss_gradient = jax.grad(get_loss)
get_loss_gradient_derivatives_wrt_pi = jax.jacrev(get_loss_gradient, 5)
get_loss_hessian = jax.hessian(get_loss)


# TODO: Add redo analysis option?
@click.command()
@click.option(
    "--study_dataframe_pickle",
    type=click.File("rb"),
    help="Pickled pandas dataframe in correct format (see contract/readme)",
)
# TODO: Eventually this will just be like the algo statistics object and
# anything else we need after parsing by intermediate package.
@click.option("--rl_algorithm_object_pickle", type=click.File("rb"))
def analyze_dataset(study_dataframe_pickle, rl_algorithm_object_pickle):
    study_df = pickle.load(study_dataframe_pickle)
    study_RLalg = pickle.load(rl_algorithm_object_pickle)

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

    # TODO: state features should not be the same as RL alg for full generality
    theta_est = estimate_theta(
        study_df, study_RLalg.state_feats, study_RLalg.treat_feats
    )

    # TODO: state features should not be the same as RL alg for full generality
    bread_matrix = form_bread_inverse_matrix(
        study_RLalg.upper_left_bread_inverse,
        study_df,
        study_RLalg.algorithm_statistics_by_calendar_t,
        update_times,
        study_RLalg.state_feats,
        study_RLalg.treat_feats,
        study_RLalg.beta_dim,
        theta_est,
    )

    meat_matrix = form_meat_matrix(
        study_df,
        theta_est,
        study_RLalg.state_feats,
        study_RLalg.treat_feats,
        update_times,
        study_RLalg.beta_dim,
        study_RLalg.algorithm_statistics_by_calendar_t,
    )

    variance = bread_matrix @ meat_matrix @ bread_matrix.T

    print(variance)


# TODO: Think about user interface.  Do they give whole estimate theta function? or simply a model
# spec and we do the fitting within some framework
# TODO: Should we specify the format of study df or allow flexibility?
def estimate_theta(study_df, state_feats, treat_feats):
    # Note that the intercept is included in the features already (col of 1s)
    linear_model = LinearRegression(fit_intercept=False)
    trimmed_df = study_df[state_feats].copy()
    for feat in treat_feats:
        trimmed_df[f"action:{feat}"] = study_df[feat] * (
            study_df["action"] - study_df["action1prob"]
        )
    linear_model.fit(trimmed_df, study_df["reward"])

    return linear_model.coef_


# TODO: doc string
def form_meat_matrix(
    study_df,
    theta_est,
    state_feats,
    treat_feats,
    update_times,
    beta_dim,
    algo_stats_dict,
):
    user_ids = study_df.user_id.unique()
    num_rows_cols = beta_dim * len(update_times) + len(theta_est)
    running_meat_matrix = np.zeros((num_rows_cols, num_rows_cols)).astype("float64")
    for user_id in user_ids:
        user_meat_vector = np.concatenate(
            [
                algo_stats_dict[t]["loss_gradients_by_user_id"][user_id]
                for t in update_times
            ]
            + [
                get_loss_gradient(
                    theta_est,
                    *get_user_states(study_df, state_feats, treat_feats, user_id),
                    actions=get_user_actions(study_df, user_id),
                    rewards=get_user_rewards(study_df, user_id),
                    action1probs=get_user_action1probs(study_df, user_id),
                )
            ],
        ).reshape(-1, 1)
        running_meat_matrix += np.outer(user_meat_vector, user_meat_vector)

    return running_meat_matrix / len(user_ids)


# TODO: Doc string
def get_user_states(study_df, state_feats, treat_feats, user_id):
    """
    Extract just the rewards for the given user in the given study_df as a
    numpy (column) vector.
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
    study_df,
    algo_stats_dict,
    update_times,
    state_feats,
    treat_feats,
    beta_dim,
    theta_est,
):
    existing_rows = upper_left_bread_inverse.shape[0]
    user_ids = study_df.user_id.unique()

    # This is useful for sweeping through the decision times between updates
    # but critically also those after the final update
    max_t = study_df.calendar_t.max()
    update_times_and_upper_limit = (
        update_times if update_times[-1] == max_t + 1 else update_times + [max_t + 1]
    )

    theta_dim = len(theta_est)

    # Begin by creating a few convenience data structures for the mixed theta/beta derivatives
    # that are most easily created for many decision times at once, whereas the following loop is
    # over update times.  We will pull appropriate quantities from here during iterations of the
    # loop.

    # This computes derivatives of the theta estimating equation wrt the action probabilities
    # vector, which importantly has an element for *every* decision time.  We will later do the
    # work to multiply these by derivatives of pi with respect to beta, thus getting the quantities
    # we really want via the chain rule, and also summing terms that correspond to the *same* betas
    # behind the scenes.
    # NOTE THAT COLUMN INDEX i CORRESPONDS TO DECISION TIME i+1!
    # NOTE that JAX treats positional args as keyword args if they are *supplied* with name=val
    # syntax.  So though supplying these arg names is a good practice for readability, but has
    # unexpected consequences in this case. Just noting this because it was tricky to debug here.
    mixed_theta_pi_loss_derivatives_by_user_id = {
        user_id: get_loss_gradient_derivatives_wrt_pi(
            theta_est,
            *get_user_states(study_df, state_feats, treat_feats, user_id),
            get_user_actions(study_df, user_id),
            get_user_rewards(study_df, user_id),
            get_user_action1probs(study_df, user_id),
        ).squeeze()
        for user_id in user_ids
    }
    # TODO: Handle missing data?
    # This simply collects the pi derivatives with respect to betas for all
    # decision times for each user, reorganizing existing data from the RL side.
    # The one complication is that we add some padding of zeros for decision
    # times before the first update to be in correspondence with the above data
    # structure.
    # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
    pi_derivatives_by_user_id = {
        user_id: np.pad(
            np.array(
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
        running_entry_holder = np.zeros((theta_dim, theta_dim))

        # This loop calculates the per-user quantities that will be
        # averaged for the final matrix entries
        for user_id in user_ids:
            # 1. We first form the outer product of the estimating equation for theta
            # and the sum of the weight gradients with respect to beta for the
            # corresponding decision times

            theta_loss_gradient = get_loss_gradient(
                theta_est,
                *get_user_states(study_df, state_feats, treat_feats, user_id),
                get_user_actions(study_df, user_id),
                get_user_rewards(study_df, user_id),
                get_user_action1probs(study_df, user_id),
            )

            weight_gradient_sum = np.zeros(beta_dim)

            # This loop iterates over decision times in slices between updates
            # to collect the right weight gradients
            for t in range(lower_t, upper_t):
                weight_gradient_sum += algo_stats_dict[t][
                    "weight_gradients_by_user_id"
                ][user_id]

            running_entry_holder += np.outer(theta_loss_gradient, weight_gradient_sum)

            # TODO: I think we now will be getting the actual pi functions... maybe this can be simplified?
            # 2. We now calculate mixed derivatives of the loss wrt theta and then beta. This piece
            # is a bit intricate; we only have the the theta loss function in terms of the pis,
            # and the  *values* of the pi derivatives wrt to betas available here, since the actual
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
            mixed_theta_pi_loss_derivative = np.matmul(
                mixed_theta_pi_loss_derivatives_by_user_id[user_id][
                    :,
                    lower_t - 1 : upper_t - 1,
                ],
                pi_derivatives_by_user_id[user_id][
                    lower_t - 1 : upper_t - 1,
                    :,
                ],
            )

            running_entry_holder += mixed_theta_pi_loss_derivative

        bottom_left_row_blocks.append(running_entry_holder / len(user_ids))

    bottom_right_hessian = sum(
        np.array(
            get_loss_hessian(
                theta_est,
                *get_user_states(study_df, state_feats, treat_feats, user_id),
                get_user_actions(study_df, user_id),
                get_user_rewards(study_df, user_id),
                get_user_action1probs(study_df, user_id),
            )
        )
        for user_id in user_ids
    ) / len(user_ids)

    return np.block(
        [
            [
                upper_left_bread_inverse,
                np.zeros((existing_rows, theta_dim)),
            ],
            [
                np.block(bottom_left_row_blocks),
                bottom_right_hessian,
            ],
        ]
    )


if __name__ == "__main__":
    analyze_dataset()  # pylint: disable=no-value-for-parameter
