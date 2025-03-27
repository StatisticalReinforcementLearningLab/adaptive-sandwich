from collections.abc import Hashable
import logging
from typing import Any

import numpy as np
import pandas as pd
import jax.numpy as jnp

import reward_definition
from rl_algorithm import RLAlgorithm
from simulation_environment import SimulationEnvironment

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

FILL_IN_COLS = ["policy_idx", "action", "prob", "reward", "quality"] + [
    "state.tod",
    "state.b.bar",
    "state.a.bar",
    "state.app.engage",
    "state.bias",
]


# TODO: Needs unit tests
def create_base_data_df(
    template_users_list: list[str],
    per_user_weeks_in_study: int,
    users_per_recruitment: int,
    num_decision_times_per_user_per_day: int,
    weeks_between_recruitments: int,
) -> pd.DataFrame:
    """
    Create starter data dataframe for the experiment, filling in what
    we can a priori.  The data dataframe will hold the collected study data, and
    will have one row for each user for each calendar decision time (WHETHER THE USER WAS ACTIVE OR
    NOT!).

    Parameters:
    template_users_list (list[str]):
        List of user prototype ids. Non-unique! The index in the list will be taken to be the
        unique user index.

    per_user_weeks_in_study (int):
        The number of weeks each user is in the study.

    users_per_recruitment (int):
        The number of users recruited per recruitment.

    num_decision_times_per_user_per_day (int):
        The number of decision times per day for each user.

    weeks_between_recruitments (int):
        The number of weeks between each user recruitment batch.

    Returns:
    pd.DataFrame: DataFrame containing all study information for all users.
    """
    num_users = len(template_users_list)
    num_decision_times_per_user = (
        per_user_weeks_in_study * 7 * num_decision_times_per_user_per_day
    )

    # Build a list of entry weeks for all users.
    weeks_of_entry = [
        weeks_between_recruitments * (user_idx // users_per_recruitment)
        for user_idx in range(num_users)
    ]
    total_num_decision_times = (
        max(weeks_of_entry) * 7 * num_decision_times_per_user_per_day
        + num_decision_times_per_user
    )

    ### data df ###
    data_dict = dict.fromkeys(
        [
            "user_idx",
            "template_user_id",
            "user_entry_decision_t",
            "user_last_decision_t",
            "calendar_decision_t",
        ]
        + FILL_IN_COLS
    )
    data_dict["user_idx"] = np.repeat(
        np.arange(len(template_users_list)), total_num_decision_times
    )
    data_dict["template_user_id"] = np.repeat(
        template_users_list, total_num_decision_times
    )
    data_dict["user_entry_decision_t"] = (
        num_decision_times_per_user_per_day
        * 7
        * np.repeat(weeks_of_entry, total_num_decision_times)
    )
    data_dict["user_last_decision_t"] = (
        data_dict["user_entry_decision_t"] + num_decision_times_per_user - 1
    )
    data_dict["calendar_decision_t"] = np.tile(
        np.arange(total_num_decision_times), num_users
    )
    data_dict["in_study"] = np.array(
        [
            entry <= calendar <= last
            for entry, calendar, last in zip(
                data_dict["user_entry_decision_t"],
                data_dict["calendar_decision_t"],
                data_dict["user_last_decision_t"],
            )
        ]
    )
    for key in FILL_IN_COLS:
        data_dict[key] = np.full(total_num_decision_times * num_users, np.nan)
    data_df = pd.DataFrame.from_dict(data_dict)

    return data_df


def get_all_in_study_data_so_far(data_df, calendar_decision_time, regex_pattern):
    result = data_df.loc[
        (data_df["calendar_decision_t"] <= calendar_decision_time)
        & (data_df["in_study"] == 1)
    ].filter(regex=(regex_pattern))
    return result.values.flatten() if result.shape[1] == 1 else result.values


def get_all_in_study_single_user_data_prior_to_decision_t(
    data_df, user_idx, calendar_decision_time, regex_pattern
):
    result = data_df.loc[
        (data_df["user_idx"] == user_idx)
        & (data_df["calendar_decision_t"] < calendar_decision_time)
        & (data_df["in_study"] == 1)
    ].filter(regex=(regex_pattern))
    return result.values.flatten() if result.shape[1] == 1 else result.values


def set_data_df_values_for_user(
    data_df,
    user_idx,
    calendar_decision_time,
    policy_idx,
    action,
    prob,
    reward,
    quality,
    alg_state,
):
    data_df.loc[
        (data_df["user_idx"] == user_idx)
        & (data_df["calendar_decision_t"] == calendar_decision_time),
        FILL_IN_COLS,
    ] = np.concatenate([[policy_idx, action, prob, reward, quality], alg_state])


# if user did not open the app at all before the decision time, then we simulate
# the algorithm selecting action based off of a stale state (i.e., b_bar is the b_bar from when the
# user last opened their app) if user did open the app, then the algorithm selecting action based
# off of a fresh state (i.e., b_bar stays the same)
def get_alg_state_from_app_opening(
    user_last_open_app_dt,
    data_df,
    user_idx,
    user_start_calendar_time,
    user_decision_time,
    advantage_state,
):
    """
    Get the algorithm state based on the time of the last app open.

    Note that we deal in both user and calendar time here, since the environment deals in user
    decision times, but the data dataframe deals in calendar decision times.

    Assumes two decision times per day.

    """

    # if morning dt we check if users opened the app in the morning
    # if evening dt we check if users opened the app in the morning and in the evening
    if user_decision_time % 2 == 0:
        user_opened_app_today = user_last_open_app_dt == user_decision_time
    else:
        # we only simulate users opening the app for morning dts
        user_opened_app_today = user_last_open_app_dt == user_decision_time - 1
    if not user_opened_app_today:
        # impute b_bar with stale b_bar and prior day app engagement = 0
        calendar_last_open_app_dt = user_start_calendar_time + user_last_open_app_dt
        stale_b_bar = get_all_in_study_single_user_data_prior_to_decision_t(
            data_df, user_idx, calendar_last_open_app_dt + 1, "state.b.bar"
        )[-1]
        # refer to rl_algorithm.py process_alg_state functions for V2, V3
        advantage_state[1] = stale_b_bar
        advantage_state[3] = 0

    return advantage_state


def get_previous_day_qualities_and_actions(user_decision_time, Qs, As):
    if user_decision_time > 1:
        if user_decision_time % 2 == 0:
            return Qs, As
        else:
            # current evening dt does not use most recent quality or action
            return Qs[:-1], As[:-1]
    # first day return empty Qs and As back
    else:
        return Qs, As


def form_beta_from_posterior(
    posterior_mean: np.ndarray, posterior_var: np.ndarray
) -> jnp.ndarray:
    """
    Form the beta vector from the posterior mean and variance.
    This is for after-study analysis, concisely collecting all the information
    in the posterior in a convenint form. Explicitly, we concatenate the posterior
    mean with the upper triangular elements of the inverse posterior variance matrix.

    Parameters:
    posterior_mean (np.ndarray):
        The posterior mean vector.
    posterior_var (np.ndarray):
        The posterior variance matrix.

    Returns:
    jnp.ndarray: The beta vector.
    """
    sigma_inv = np.linalg.inv(posterior_var)
    ut_sigma_inv = sigma_inv[np.triu_indices_from(sigma_inv)]
    return jnp.concatenate([posterior_mean, ut_sigma_inv.flatten()])


def execute_decision_time(
    data_df: pd.DataFrame,
    user_idx: int,
    calendar_decision_time: int,
    algorithm: RLAlgorithm,
    sim_env: SimulationEnvironment,
    policy_idx: int,
    num_decision_times_per_user_per_day: int,
    per_user_weeks_in_study: int,
    action_prob_function_args: dict,
):
    """
    Execute a decision time for a user, updating the data dataframe with the results.

    Parameters:
    data_df (pd.DataFrame):
        The data dataframe containing all study information for all users.  This dataframe will be
        updated with the results of the decision time.
    user_idx (int):
        The index of the user for whom the decision time is being executed. Recall that this is
        different from the user id of the corresponding template user.
    calendar_decision_time (int):
        The calendar decision time.
    algorithm (RLAlgorithm):
        The algorithm object that contains methods for action selection and periodically updating
        based on results so far. The action selection method is the main thing used here.
    sim_env (SimulationEnvironment):
        The simulation environment object that generates the context for the experiment. This is
        mainly used to generate user states and rewards here.
    policy_idx (int):
        The index of the policy being used for this decision time.
    num_decision_times_per_user_per_day (int):
        The number of decision times per day for each user. This is assumed to be 2 for now.
    per_user_weeks_in_study (int):
        The number of weeks each user is in the study.
    action_prob_function_args (dict):
        A dictionary of arguments to be passed to the version of the action probability function.
        on the after-study-analysis side.  Note this is a slightly different function than the
        one used here for JAX-differentiability reasons.

    Returns:
    None
    """
    # The environment deals in user decision times, starting at 0 for each person
    # We entertain that instead of refactoring, though I prefer dealing in calendar
    # time outisde of the environment.
    user_start_time = data_df.loc[(data_df["user_idx"] == user_idx)][
        "user_entry_decision_t"
    ].values[0]
    user_decision_time = calendar_decision_time - user_start_time
    env_state = sim_env.generate_current_state(
        user_idx, user_decision_time, per_user_weeks_in_study
    )

    user_qualities = get_all_in_study_single_user_data_prior_to_decision_t(
        data_df, user_idx, calendar_decision_time, "quality"
    )
    user_actions = get_all_in_study_single_user_data_prior_to_decision_t(
        data_df, user_idx, calendar_decision_time, "action"
    )

    Qs, As = get_previous_day_qualities_and_actions(
        user_decision_time, user_qualities, user_actions
    )
    b_bar, a_bar = reward_definition.get_b_bar_a_bar(Qs, As)
    # The second return is "baseline" state, not used here.  We pass the advantage
    # state to the action selection function as both the advantage and baseline state,
    # but that baseline state arg is also not used.
    advantage_state, _ = algorithm.process_alg_state(env_state, b_bar, a_bar)

    # Simulate app opening issue (from a state development perspective, not a policy overlap
    # perspective)
    if sim_env.get_version() == "V2" or sim_env.get_version() == "V3":
        # Grab the last USER decision time the user opened the app.
        user_last_open_app_dt = sim_env.get_user_last_open_app_dt(user_idx)
        alg_state = get_alg_state_from_app_opening(
            user_last_open_app_dt,
            data_df,
            user_idx,
            user_start_time,
            user_decision_time,
            advantage_state,
        )
    else:
        alg_state = advantage_state

    action, action_prob = algorithm.action_selection(
        advantage_state=alg_state, baseline_state=alg_state
    )
    action_prob_function_args[calendar_decision_time][user_idx] = (
        form_beta_from_posterior(algorithm.posterior_mean, algorithm.posterior_var),
        jnp.array(alg_state),
        algorithm.feature_dim,
    )

    quality = sim_env.generate_rewards(user_idx, env_state, action)
    reward = algorithm.reward_def_func(quality, action, b_bar, a_bar)

    set_data_df_values_for_user(
        data_df,
        user_idx,
        calendar_decision_time,
        policy_idx,
        action,
        action_prob,
        reward,
        quality,
        alg_state,
    )
    # Update responsiveness if after the first week
    if calendar_decision_time >= 7 * num_decision_times_per_user_per_day:
        sim_env.update_responsiveness(
            user_idx,
            reward_definition.calculate_a1_condition(a_bar),
            reward_definition.calculate_a2_condition(a_bar),
            reward_definition.calculate_b_condition(b_bar),
            calendar_decision_time,
        )


def run_incremental_recruitment_exp(
    template_users_list: list[str],
    users_per_recruitment: int,
    algorithm: RLAlgorithm,
    sim_env: SimulationEnvironment,
    per_user_weeks_in_study: int,
    num_users_before_update: int,
    num_decision_times_per_user_per_day: int,
    weeks_between_recruitments: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[Hashable, tuple[Any, ...]]],
    dict[int, dict[Hashable, tuple[Any, ...]]],
]:
    """
    Run an incremental recruitment experiment where treatment is applied via the supplied algorithm
    and user data is generated by the supplied simulation environment.

    Parameters:
    template_users_list (list[str]):
        List of user prototype ids. Non-unique! The index in the list will be taken to be the unique
        user index.

    users_per_recruitment (int):
        The number of new users recruited per recruitment.

    algorithm (RLAlgorithm):
        The algorithm object that contains methods for decision making and updating based on the
        experiment's results.

    sim_env (SimulationEnvironment):
        The simulation environment object that generates the context for the experiment.

    per_user_weeks_in_study (int):
        The number of weeks each user is in the study.

    num_users_before_update (int):
        The number of users required before updating the algorithm.

    num_decision_times_per_user_per_day (int):
        The number of decision times per day for each user.

    weeks_between_recruitments (int):
        The number of weeks between each user recruitment batch.


    Returns:
    tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[Hashable, tuple[Any, ...]]], dict[int, dict[Hashable, tuple[Any, ...]]]]: A tuple containing two DataFrames:
        - data_df: DataFrame containing the data collected during the experiment.
        - update_df: DataFrame containing the updates to the algorithm's posterior mean and variance.
        - alg_update_function_args: A dictionary keyed on update index and user id, containting the
            arguments to be passed to the version of the algorithm update function that is used for
            after-study analysis.
        - action_prob_function_args: A dictionary keyed on calendar decision time and user id,
            containing the arguments to be passed to the version of the action probability function
            that is used for after-study analysis.
    """

    if num_decision_times_per_user_per_day != 2:
        raise ValueError(
            "This function assumes two decision times per day currently. Please update all relevant"
            "logic if you'd like to change this."
        )

    data_df = create_base_data_df(
        template_users_list,
        per_user_weeks_in_study,
        users_per_recruitment,
        num_decision_times_per_user_per_day,
        weeks_between_recruitments,
    )
    # Initialize the dictionaries that will collect function arguments needed for after-study
    # analysis.
    alg_update_function_args = {}
    action_prob_function_args = {}

    # Begin with prior values in update dict
    update_dict = {}
    policy_idx = 0
    update_dict[policy_idx] = np.concatenate(
        [algorithm.PRIOR_MU, algorithm.PRIOR_SIGMA.flatten()]
    )

    # The core loop.
    # Loop over all decision times, recruiting and retiring users as dictated by
    # users_per_recruitment and per_user_weeks_in_study, and updating the algorithm when the pure
    # exploration has ended and the cadence for updates has been reached.
    num_decision_times_between_updates = algorithm.update_cadence
    for calendar_t in range(data_df.calendar_decision_t.max() + 1):
        action_prob_function_args[calendar_t] = {}
        logger.info("Simulating calendar decision time: %s", calendar_t)
        for user_idx in range(len(template_users_list)):
            filtered_df_row = data_df[
                (data_df.user_idx == user_idx)
                & (data_df.calendar_decision_t == calendar_t)
            ].squeeze()
            if filtered_df_row.in_study:
                # Perform reward generation, action selection, record the results in data_df, and
                # collect the corresponding action probability function args for after-study
                # analysis.
                execute_decision_time(
                    data_df,
                    user_idx,
                    filtered_df_row.calendar_decision_t,
                    algorithm,
                    sim_env,
                    policy_idx,
                    num_decision_times_per_user_per_day,
                    per_user_weeks_in_study,
                    action_prob_function_args,
                )
            else:
                action_prob_function_args[calendar_t][user_idx] = ()

        # Check if an update is potentially warranted.
        if (calendar_t < data_df.calendar_decision_t.max()) and (
            calendar_t + 1
        ) % num_decision_times_between_updates == 0:
            # Check if we have enough users to end the pure exploration period.
            if (
                algorithm.is_pure_exploration_period()
                and data_df[
                    (data_df.in_study == 1)
                    & (data_df.calendar_decision_t <= calendar_t)
                ].user_idx.nunique()
                >= num_users_before_update
            ):
                algorithm.end_pure_exploration_period()

            # If still pure exploring, skip the update.
            if algorithm.is_pure_exploration_period():
                logger.info(
                    "Skipping update, as conditions for ending pure exploration not met."
                )
                continue

            # Update the algorithm and collect the args to the corresponding loss/estimating
            # function for after-study analysis.
            logger.info("Updating algorithm.")
            alg_states = get_all_in_study_data_so_far(data_df, calendar_t, "state.*")
            actions = get_all_in_study_data_so_far(data_df, calendar_t, "action")
            pis = get_all_in_study_data_so_far(data_df, calendar_t, "prob")
            rewards = get_all_in_study_data_so_far(data_df, calendar_t, "reward")
            algorithm.update(alg_states, actions, pis, rewards)
            policy_idx += 1
            update_dict[policy_idx] = np.concatenate(
                [algorithm.posterior_mean, algorithm.posterior_var.flatten()]
            )
            update_alg_update_function_args(
                alg_update_function_args, policy_idx, data_df, algorithm
            )

    update_data = [
        [update_idx] + update_dict[update_idx].tolist()
        for update_idx in sorted(update_dict.keys())
    ]
    columns = (
        ["update_idx"]
        + [f"posterior_mu.{i}" for i in range(algorithm.feature_dim)]
        + [
            f"posterior_var.{i}.{j}"
            for i in range(algorithm.feature_dim)
            for j in range(algorithm.feature_dim)
        ]
    )
    update_df = pd.DataFrame(update_data, columns=columns)

    study_df = create_study_df(data_df)

    return (
        data_df,
        update_df,
        study_df,
        alg_update_function_args,
        action_prob_function_args,
    )


def create_study_df(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the study dataframe, a T x n pandas dataframe where T is the set of all calendar decision
    times for which at least one user is active and n is the total number of users in the study.

    Really this is a minor refactoring of the data_df to be compatible with the after-study analysis
    package.  We could just build this directly instead of building data_df, but not going to do
    that surgery for now.

    Parameters:
    data_df (pd.DataFrame):
        The data dataframe containing all study information for all users.

    Returns:
        pd.DataFrame: The study dataframe.
    """

    study_df = pd.DataFrame(
        columns=[
            "calendar_decision_t",
            "user_idx",
            "in_study_indicator",
            "action",
            "policy_idx",
            "act_prob",
            "reward",
            "oscb",
            "tod",
            "bbar",
            "abar",
            "appengage",
            "bias",
        ]
    )

    for calendar_decision_t in range(
        data_df["calendar_decision_t"].min(), data_df["calendar_decision_t"].max() + 1
    ):
        temp = data_df[
            data_df["calendar_decision_t"] == calendar_decision_t
        ].reset_index(drop=True)

        for user in data_df["user_idx"].unique():
            record = temp[temp["user_idx"] == user].reset_index(drop=True)

            # Check if the user is in the study
            if record.in_study.item():
                # Fetch the policy number and associated beta
                policy = record["policy_idx"].values[0]

                # Fetch the state
                state = jnp.array(
                    record[
                        [
                            "state.tod",
                            "state.b.bar",
                            "state.a.bar",
                            "state.app.engage",
                            "state.bias",
                        ]
                    ].values[0],
                    dtype=np.float32,
                )

                study_df = pd.concat(
                    [
                        study_df,
                        pd.DataFrame(
                            [
                                [
                                    calendar_decision_t,
                                    user,
                                    1,
                                    record["action"].values[0],
                                    policy,
                                    record["prob"].values[0],
                                    record["reward"].values[0],
                                    record["quality"].values[0],
                                    state[0].item(),
                                    state[1].item(),
                                    state[2].item(),
                                    state[3].item(),
                                    state[4].item(),
                                ]
                            ],
                            columns=[
                                "calendar_decision_t",
                                "user_idx",
                                "in_study_indicator",
                                "action",
                                "policy_idx",
                                "act_prob",
                                "reward",
                                "oscb",
                                "tod",
                                "bbar",
                                "abar",
                                "appengage",
                                "bias",
                            ],
                        ),
                    ]
                )
            else:
                study_df = pd.concat(
                    [
                        study_df,
                        pd.DataFrame(
                            [
                                [
                                    calendar_decision_t,
                                    user,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                ]
                            ],
                            columns=[
                                "calendar_decision_t",
                                "user_idx",
                                "in_study_indicator",
                                "action",
                                "policy_idx",
                                "act_prob",
                                "reward",
                                "oscb",
                                "tod",
                                "bbar",
                                "abar",
                                "appengage",
                                "bias",
                            ],
                        ),
                    ]
                )
    study_df.reset_index(drop=True, inplace=True)
    study_df = study_df.infer_objects()

    return study_df


def update_alg_update_function_args(
    alg_update_function_args: dict[int, dict[Hashable, tuple[Any, ...]]],
    policy_idx: int,
    data_df: pd.DataFrame,
    algorithm: RLAlgorithm,
) -> None:
    """
    Update the algorithm update function arguments dictionary with the arguments for the current
    policy update.

    Parameters:
    alg_update_function_args (dict[int, dict[Hashable, tuple[Any, ...]]]):
        The dictionary of arguments to be passed to the version of the algorithm update function that
        is used for after-study analysis. The dictionary is keyed on update index and user id.
    policy_idx (int):
        The index of the policy produced by the update just executed.

    data_df (pd.DataFrame):
        The data dataframe containing all study information for all users.

    algorithm (RLAlgorithm):
        The algorithm object that contains the posterior mean and variance for the current policy.

    """
    prior_mu = algorithm.PRIOR_MU
    prior_sigma_inv = np.linalg.inv(algorithm.PRIOR_SIGMA)

    beta = form_beta_from_posterior(algorithm.posterior_mean, algorithm.posterior_var)

    num_users_entered_already = data_df[
        (data_df["policy_idx"] < policy_idx) & (data_df["in_study"] == 1)
    ]["user_idx"].nunique()
    alg_update_function_args[policy_idx] = {}
    for user_idx in data_df["user_idx"].unique():
        # Create the data dataframe
        temp = data_df[
            (data_df["policy_idx"] < policy_idx)
            & (data_df["user_idx"] == user_idx)
            & (data_df["in_study"] == 1)
        ].reset_index(drop=True)

        # Check if the user has any data
        if temp.shape[0] != 0:
            # Sort by calendar_decision_t
            temp = temp.sort_values(by="calendar_decision_t")

            states = []
            actions = []
            act_probs = []
            decision_times = []
            rewards = jnp.array(temp["reward"].values)
            for j in range(temp.shape[0]):
                state = np.array(
                    temp.loc[j][
                        [
                            "state.tod",
                            "state.b.bar",
                            "state.a.bar",
                            "state.app.engage",
                            "state.bias",
                        ]
                    ].values,
                    dtype=np.float32,
                )
                action = temp.loc[j]["action"]
                act_prob = temp.loc[j]["prob"]
                decision_time = temp.loc[j]["calendar_decision_t"]

                states.append(state)
                actions.append(action)
                act_probs.append(act_prob)
                decision_times.append(decision_time)

            states = jnp.array(states)
            actions = jnp.array(actions).reshape(-1, 1)
            act_probs = jnp.array(act_probs).reshape(-1, 1)
            decision_times = jnp.array(decision_times).reshape(-1, 1)

            alg_update_function_args[policy_idx][user_idx] = (
                beta,
                num_users_entered_already,
                states,
                actions,
                act_probs,
                decision_times,
                rewards,
                prior_mu,
                prior_sigma_inv,
                3396.449,
            )

        else:
            alg_update_function_args[policy_idx][user_idx] = ()
