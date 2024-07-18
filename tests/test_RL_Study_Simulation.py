import argparse
import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import synthetic_env
import basic_RL_algorithms
import rl_study_simulation
import constants


class TestRunStudySimulation:
    def setup_method(self):
        """
        Runs anew before each test
        """
        self.args_incremental_1 = argparse.Namespace(
            dataset_type="synthetic",
            verbose=0,
            synthetic_mode="delayed_1_dosage",
            RL_alg="sigmoid_LS",
            N=1,
            n=6,
            recruit_n=2,
            upper_clip=0.9,
            lower_clip=0.1,
            fixed_action_prob=0.5,
            min_users=1,
            err_corr="time_corr",
            decisions_between_updates=2,
            save_dir=".",
            steepness=1.0,
            alg_state_feats="intercept,past_reward",
            action_centering=0,
            prior="naive",
            T=4,
            recruit_t=1,
            allocation_sigma=1,
            noise_var=1,
        )

        self.args_no_incremental_1 = argparse.Namespace(
            dataset_type="synthetic",
            verbose=0,
            synthetic_mode="delayed_1_dosage",
            RL_alg="sigmoid_LS",
            N=1,
            n=6,
            recruit_n=6,
            upper_clip=0.9,
            lower_clip=0.1,
            fixed_action_prob=0.5,
            min_users=1,
            err_corr="time_corr",
            decisions_between_updates=2,
            save_dir=".",
            steepness=1.0,
            alg_state_feats="intercept,past_reward",
            action_centering=0,
            prior="naive",
            T=4,
            recruit_t=1,
            allocation_sigma=1,
            noise_var=1,
        )

        self.state_feats = [
            constants.RLStudyArgs.INTERCEPT,
            constants.RLStudyArgs.PAST_REWARD,
        ]
        self.treat_feats = self.state_feats
        self.sigmoid_1 = basic_RL_algorithms.SigmoidLS(
            state_feats=self.state_feats,
            treat_feats=self.treat_feats,
            alg_seed=int(time.time()),
            steepness=self.args_incremental_1.steepness,
            lower_clip=self.args_incremental_1.lower_clip,
            upper_clip=self.args_incremental_1.upper_clip,
            action_centering=self.args_incremental_1.action_centering,
        )
        self.sigmoid_1.rng = MagicMock(autospec=True)
        self.sigmoid_1.get_action_probs = MagicMock(autospec=True)

        # Generation features
        past_action_len = 1
        past_action_cols = [constants.RLStudyArgs.INTERCEPT] + [
            f"past_action_{i}" for i in range(1, past_action_len + 1)
        ]
        past_reward_action_cols = ["past_reward"] + [
            f"past_action_{i}_reward" for i in range(1, past_action_len + 1)
        ]
        gen_feats = past_action_cols + past_reward_action_cols + ["dosage"]

        self.study_env_1 = synthetic_env.SyntheticEnv(
            self.args_incremental_1,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            env_seed=int(time.time()),
            gen_feats=gen_feats,
            err_corr=self.args_incremental_1.err_corr,
        )
        self.study_env_1.rng = MagicMock(autospec=True)
        self.study_env_1.rng = MagicMock(autospec=True)
        self.study_env_1.sample_rewards = MagicMock(
            spec=synthetic_env.SyntheticEnv.sample_rewards
        )

    def test_run_study_simulation_incremental_recruitment(self):
        self.sigmoid_1.get_action_probs.side_effect = [
            np.array([0.6, 0.3]),
            np.array([0.6, 0.3]),
            np.array([0.6, 0.3, 0.6, 0.3]),
            np.array([0.6, 0.3, 0.6, 0.3]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.6, 0.3, 0.6, 0.3]),
            np.array([0.6, 0.3]),
            np.array([0.6, 0.3]),
        ]
        self.sigmoid_1.rng.binomial.side_effect = [
            np.array([1, 0]),
            np.array([1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([0, 0, 1, 0]),
            np.array([1, 1, 0, 1]),
            np.array([1, 0]),
            np.array([1, 0]),
        ]
        self.study_env_1.sample_rewards.side_effect = [
            np.array([0.1, 0.9]),
            np.array([0.8, 0.6]),
            np.array([1.0, 0, 1.0, 0]),
            np.array([0.5, 1.5, -1, 0]),
            np.array([0.5, 1.5, -1, 0]),
            np.array([-0.5, 2.3, -1, 0.9]),
            np.array([0.0, 0.0]),
            np.array([10, 4.67]),
        ]
        self.study_env_1.rng.normal.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        study_df, _ = rl_study_simulation.run_study_simulation(
            args=self.args_incremental_1,
            study_env=self.study_env_1,
            study_RLalg=self.sigmoid_1,
            user_env_data=None,
        )

        expected_df = pd.DataFrame(
            {
                "user_id": np.repeat(np.arange(1, 7), 8),
                "policy_num": [
                    # USER 1
                    1,
                    1,
                    2,
                    2,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    1,
                    1,
                    2,
                    2,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    2,
                    2,
                    3,
                    3,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    2,
                    2,
                    3,
                    3,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    3,
                    3,
                    4,
                    4,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    3,
                    3,
                    4,
                    4,
                ],
                "last_t": np.repeat([4, 6, 8], 16),
                "entry_t": np.repeat([1, 3, 5], 16),
                "calendar_t": np.tile(np.arange(1, 9), 6),
                "action1prob": [
                    # USER 1
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    0.3,
                    0.3,
                    0.3,
                    0.3,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    0.6,
                    0.6,
                    0.5,
                    0.6,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    0.3,
                    0.3,
                    0.5,
                    0.3,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    0.5,
                    0.6,
                    0.6,
                    0.6,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    0.5,
                    0.3,
                    0.3,
                    0.3,
                ],
                "intercept": [1] * 48,
                "action": [
                    # USER 1
                    1,
                    1,
                    1,
                    1,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    1,
                    1,
                    0,
                    1,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    0,
                    0,
                    0,
                    1,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    1,
                    0,
                    1,
                    1,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    0,
                    1,
                    0,
                    0,
                ],
                "reward": [
                    # USER 1
                    0.1,
                    0.8,
                    1.0,
                    0.5,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    0.9,
                    0.6,
                    0,
                    1.5,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    1.0,
                    -1,
                    0.5,
                    -0.5,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    0,
                    0,
                    1.5,
                    2.3,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    -1,
                    -1,
                    0.0,
                    10,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    0,
                    0.9,
                    0.0,
                    4.67,
                ],
                "past_action_1": [
                    # USER 1
                    0,
                    1,
                    1,
                    1,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    0,
                    1,
                    1,
                    0,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    0,
                    1,
                    0,
                    1,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    1,
                    0,
                ],
                "past_reward": [
                    # USER 1
                    0.1,
                    0.1,
                    0.8,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    0.2,
                    0.9,
                    0.6,
                    0,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    0.3,
                    1.0,
                    -1,
                    0.5,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    0.4,
                    0,
                    0,
                    1.5,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    0.5,
                    -1,
                    -1,
                    0.0,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    0.6,
                    0,
                    0.9,
                    0.0,
                ],
                "past_action_1_reward": [
                    # USER 1
                    0,
                    0.1,
                    0.8,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    0,
                    1.0,
                    -1,
                    0,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    0,
                    -1,
                    0,
                    0.0,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    0.9,
                    0,
                ],
                "dosage": [
                    # USER 1
                    0.0,
                    0.050000000000000044,
                    0.09750000000000009,
                    0.14262500000000014,
                    None,
                    None,
                    None,
                    None,
                    # USER 2
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    # USER 3
                    None,
                    None,
                    0.0,
                    0.050000000000000044,
                    0.09750000000000009,
                    0.09262500000000007,
                    None,
                    None,
                    # USER 4
                    None,
                    None,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    None,
                    None,
                    # USER 5
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    0.050000000000000044,
                    0.04750000000000004,
                    0.09512500000000007,
                    # USER 6
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    0.0,
                    0.050000000000000044,
                    0.04750000000000004,
                ],
                "in_study": [
                    # USER 1
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    # USER 2
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    # USER 3
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    # USER 4
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    # USER 5
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    # USER 6
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                ],
                "in_study_row_index": [
                    # USER 1
                    0,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    # USER 2
                    4,
                    5,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    # USER 3
                    7,
                    7,
                    8,
                    9,
                    10,
                    11,
                    11,
                    11,
                    # USER 4
                    11,
                    11,
                    12,
                    13,
                    14,
                    15,
                    15,
                    15,
                    # USER 5
                    15,
                    15,
                    15,
                    15,
                    16,
                    17,
                    18,
                    19,
                    # USER 6
                    19,
                    19,
                    19,
                    19,
                    20,
                    21,
                    22,
                    23,
                ],
            }
        )
        expected_df = expected_df.astype({"policy_num": "float64"})
        pd.testing.assert_frame_equal(study_df, expected_df)

    def test_run_study_simulation_no_incremental_recruitment(self):
        self.sigmoid_1.get_action_probs.side_effect = [
            np.array([0.6, 0.3, 0.6, 0.3, 0.5, 0.5]),
            np.array([0.6, 0.3, 0.6, 0.3, 0.6, 0.3]),
            np.array([0.6, 0.3, 0.5, 0.5, 0.6, 0.3]),
            np.array([0.6, 0.3, 0.6, 0.3, 0.6, 0.3]),
        ]
        self.sigmoid_1.rng.binomial.side_effect = [
            np.array([1, 0, 1, 0, 1, 0]),
            np.array([1, 0, 1, 0, 0, 1]),
            np.array([1, 0, 0, 0, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0]),
        ]
        self.study_env_1.sample_rewards.side_effect = [
            np.array([0.1, 0.9, 1.0, 0, -1, 0]),
            np.array([0.8, 0.6, -1, 0, -1, 0.9]),
            np.array([1.0, 0, 0.5, 1.5, 0.0, 0.0]),
            np.array([0.5, 1.5, -0.5, 2.3, 10, 4.67]),
        ]
        self.study_env_1.rng.normal.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        study_df, _ = rl_study_simulation.run_study_simulation(
            args=self.args_no_incremental_1,
            study_env=self.study_env_1,
            study_RLalg=self.sigmoid_1,
            user_env_data=None,
        )
        expected_df = pd.DataFrame(
            {
                "user_id": np.repeat(np.arange(1, 7), 4),
                "policy_num": np.tile([1, 1, 2, 2], 6),
                "last_t": [4] * 24,
                "entry_t": [1] * 24,
                "calendar_t": np.tile(np.arange(1, 5), 6),
                "action1prob": [
                    # USER 1
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    # USER 2
                    0.3,
                    0.3,
                    0.3,
                    0.3,
                    # USER 3
                    0.6,
                    0.6,
                    0.5,
                    0.6,
                    # USER 4
                    0.3,
                    0.3,
                    0.5,
                    0.3,
                    # USER 5
                    0.5,
                    0.6,
                    0.6,
                    0.6,
                    # USER 6
                    0.5,
                    0.3,
                    0.3,
                    0.3,
                ],
                "intercept": [1] * 24,
                "action": [
                    # USER 1
                    1,
                    1,
                    1,
                    1,
                    # USER 2
                    0,
                    0,
                    0,
                    0,
                    # USER 3
                    1,
                    1,
                    0,
                    1,
                    # USER 4
                    0,
                    0,
                    0,
                    0,
                    # USER 5
                    1,
                    0,
                    1,
                    1,
                    # USER 6
                    0,
                    1,
                    0,
                    0,
                ],
                "reward": [
                    # USER 1
                    0.1,
                    0.8,
                    1.0,
                    0.5,
                    # USER 2
                    0.9,
                    0.6,
                    0,
                    1.5,
                    # USER 3
                    1.0,
                    -1,
                    0.5,
                    -0.5,
                    # USER 4
                    0,
                    0,
                    1.5,
                    2.3,
                    # USER 5
                    -1,
                    -1,
                    0.0,
                    10,
                    # USER 6
                    0,
                    0.9,
                    0.0,
                    4.67,
                ],
                "past_action_1": [
                    # USER 1
                    0,
                    1,
                    1,
                    1,
                    # USER 2
                    0,
                    0,
                    0,
                    0,
                    # USER 3
                    0,
                    1,
                    1,
                    0,
                    # USER 4
                    0,
                    0,
                    0,
                    0,
                    # USER 5
                    0,
                    1,
                    0,
                    1,
                    # USER 6
                    0,
                    0,
                    1,
                    0,
                ],
                "past_reward": [
                    # USER 1
                    0.1,
                    0.1,
                    0.8,
                    1.0,
                    # USER 2
                    0.2,
                    0.9,
                    0.6,
                    0,
                    # USER 3
                    0.3,
                    1.0,
                    -1,
                    0.5,
                    # USER 4
                    0.4,
                    0,
                    0,
                    1.5,
                    # USER 5
                    0.5,
                    -1,
                    -1,
                    0.0,
                    # USER 6
                    0.6,
                    0,
                    0.9,
                    0.0,
                ],
                "past_action_1_reward": [
                    # USER 1
                    0,
                    0.1,
                    0.8,
                    1.0,
                    # USER 2
                    0,
                    0,
                    0,
                    0,
                    # USER 3
                    0,
                    1.0,
                    -1,
                    0,
                    # USER 4
                    0,
                    0,
                    0,
                    0,
                    # USER 5
                    0,
                    -1,
                    0,
                    0.0,
                    # USER 6
                    0,
                    0,
                    0.9,
                    0,
                ],
                "dosage": [
                    # USER 1
                    0.0,
                    0.050000000000000044,
                    0.09750000000000009,
                    0.14262500000000014,
                    # USER 2
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    # USER 3
                    0.0,
                    0.050000000000000044,
                    0.09750000000000009,
                    0.09262500000000007,
                    # USER 4
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    # USER 5
                    0.0,
                    0.050000000000000044,
                    0.04750000000000004,
                    0.09512500000000007,
                    # USER 6
                    0.0,
                    0.0,
                    0.050000000000000044,
                    0.04750000000000004,
                ],
                "in_study": [1] * 24,
                "in_study_row_index": np.arange(24),
            }
        )
        expected_df = expected_df.astype(
            {"policy_num": "float64", "action": "float64", "past_action_1": "float64"}
        )
        pd.testing.assert_frame_equal(study_df, expected_df)


@pytest.mark.skip(reason="To be implemented")
def test_load_data_and_simulate_studies():
    raise NotImplementedError()
