import logging

import numpy as np
import pandas as pd

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
DECISIONS_PER_DAY = 2
WEEKS_BETWEEN_RECRUITMENTS = 2


def create_base_data_df(
    template_users_list: list[str],
    per_user_weeks_in_trial: int,
    users_per_recruitment: int,
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

    per_user_weeks_in_trial (int):
        The number of weeks each user is in the study.

    users_per_recruitment (int):
        The number of users recruited per recruitment.

    Returns:
    pd.DataFrame: DataFrame containing all study information for all users.
    """
    num_users = len(template_users_list)
    num_decision_times_per_user = per_user_weeks_in_trial * 7 * DECISIONS_PER_DAY

    # Build a list of entry weeks for all users.  Note that we recruit users
    # every two weeks, so that we may operate at the week level here.
    weeks_of_entry = [
        WEEKS_BETWEEN_RECRUITMENTS * (user_idx // users_per_recruitment)
        for user_idx in range(num_users)
    ]
    total_num_decision_times = (
        max(weeks_of_entry) * 7 * DECISIONS_PER_DAY + num_decision_times_per_user
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
        WEEKS_BETWEEN_RECRUITMENTS
        * 7
        * DECISIONS_PER_DAY
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

    """
    if DECISIONS_PER_DAY != 2:
        raise ValueError("This function is only implemented for 2 decisions per day")

    # if morning dt we check if users opened the app in the morning
    # if evening dt we check if users opened the app in the morning and in the evening
    if user_decision_time % DECISIONS_PER_DAY == 0:
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
    if DECISIONS_PER_DAY != 2:
        raise ValueError("This function is only implemented for 2 decisions per day")

    if user_decision_time > 1:
        if user_decision_time % 2 == 0:
            return Qs, As
        else:
            # current evening dt does not use most recent quality or action
            return Qs[:-1], As[:-1]
    # first day return empty Qs and As back
    else:
        return Qs, As


def execute_decision_time(
    data_df, user_idx, calendar_decision_time, algorithm, sim_env, policy_idx
):
    # The environment deals in user decision times, starting at 0 for each person
    # We entertain that instead of refactoring, though I prefer dealing in calendar
    # time outisde of the environment.
    user_start_time = data_df.loc[(data_df["user_idx"] == user_idx)][
        "user_entry_decision_t"
    ].values[0]
    user_decision_time = calendar_decision_time - user_start_time
    env_state = sim_env.generate_current_state(user_idx, user_decision_time)

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
    advantage_state, _ = algorithm.process_alg_state(env_state, b_bar, a_bar)

    # Simulate app opening issue (from a state development perspective, not a policy version
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
    if calendar_decision_time >= 7 * DECISIONS_PER_DAY:
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
    per_user_weeks_in_trial: int,
    num_users_before_update: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    per_user_weeks_in_trial (int):
        The number of weeks each user is in the study.

    num_users_before_update (int):
        The number of users required before updating the algorithm.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
        - data_df: DataFrame containing the data collected during the experiment.
        - update_df: DataFrame containing the updates to the algorithm's posterior mean and variance.
    """
    data_df = create_base_data_df(
        template_users_list,
        per_user_weeks_in_trial,
        users_per_recruitment,
    )
    # Begin with prior values in update dict
    update_dict = {}
    policy_idx = 0
    update_dict[policy_idx] = np.concatenate(
        [algorithm.PRIOR_MU, algorithm.PRIOR_SIGMA.flatten()]
    )

    # The core loop.
    # Loop over all decision times, recruiting and retiring users as dictated by
    # users_per_recruitment and per_user_weeks_in_trial, and updating the algorithm when the pure
    # exploration has ended and the cadence for updates has been reached.
    num_decision_times_between_updates = algorithm.update_cadence
    for calendar_t in range(data_df.calendar_decision_t.max() + 1):
        logger.info("Simulating calendar decision time: %s", calendar_t)
        for user_idx in range(len(template_users_list)):
            filtered_df_row = data_df[
                (data_df.user_idx == user_idx)
                & (data_df.calendar_decision_t == calendar_t)
            ].squeeze()
            if filtered_df_row.in_study:
                execute_decision_time(
                    data_df,
                    user_idx,
                    filtered_df_row.calendar_decision_t,
                    algorithm,
                    sim_env,
                    policy_idx,
                )

            # Check if an update is warranted.
        if calendar_t > 0 and calendar_t % num_decision_times_between_updates == 0:
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

            # Update the algorithm.
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

    return data_df, update_df
