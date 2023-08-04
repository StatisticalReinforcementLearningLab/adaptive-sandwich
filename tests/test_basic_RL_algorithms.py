import argparse

import pandas as pd
import numpy as np

import basic_RL_algorithms
import constants


class TestSigmoidLS_T3_n2:
    args_1 = argparse.Namespace(
        dataset_type="synthetic",
        verbose=0,
        heartsteps_mode="medium",
        synthetic_mode="delayed_1_dosage",
        RL_alg="sigmoid_LS",
        N=5,
        n=2,
        upper_clip=0.9,
        lower_clip=0.1,
        fixed_action_prob=0.5,
        min_users=1,
        err_corr="time_corr",
        decisions_between_updates=1,
        save_dir=".",
        steepness=1.0,
        alg_state_feats="intercept,past_reward",
        action_centering=0,
        prior="naive",
        T=3,
        recruit_n=50,
        recruit_t=1,
        allocation_sigma=1,
        noise_var=1,
    )
    state_feats_1 = [constants.RLStudyArgs.INTERCEPT, constants.RLStudyArgs.PAST_REWARD]
    treat_feats_1 = state_feats_1
    sigmoid_1 = basic_RL_algorithms.SigmoidLS(
        args_1,
        state_feats_1,
        treat_feats_1,
        alg_seed=1,
        allocation_sigma=args_1.allocation_sigma,
        steepness=args_1.steepness,
    )
    sigmoid_1.all_policies.append(
        {
            "beta_est": pd.DataFrame(
                {
                    "intercept": [-1.0],
                    "past_reward": [1.0],
                    "action:intercept": [2.0],
                    "action:past_reward": [3.0],
                }
            ),
            "seen_user_id": {1, 2},
        }
    )
    sigmoid_1.all_policies.append(
        {
            "beta_est": pd.DataFrame(
                {
                    "intercept": [-1.0],
                    "past_reward": [2.0],
                    "action:intercept": [3.0],
                    "action:past_reward": [4.0],
                }
            ),
            "seen_user_id": {1, 2},
        }
    )

    study_df_1 = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "calendar_t": [1, 2, 3, 1, 2, 3],
            "action": [0, 1, 1, 1, 1, 0],
            "reward": [1.0, -1, 0, 1, 0, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 1, 1, 0],
        }
    )

    def calculate_pi_gradient_manually(self):
        pass

    def calculate_weight_gradient_manually(self):
        pass

    def test_calculate_pi_and_weight_gradients(self):
        self.sigmoid_1.calculate_pi_and_weight_gradients(self.study_df_1, 3)
        breakpoint()
        np.testing.assert_equal(
            self.sigmoid_1.algorithm_statistics_by_calendar_t,
            {
                3: {
                    "pi_gradients_by_user_id": {
                        1: np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                        2: np.array([0, 0, -0.65795271, -0.0]),
                    },
                    "weight_gradients_by_user_id": {
                        1: np.array([1, 2, 3, 4]),
                        2: np.array([1, 2, 3, 4]),
                    },
                }
            },
        )

    def test_construct_upper_left_bread_matrix(self):
        self.sigmoid_1.algorithm_statistics_by_calendar_t = {
            2: {
                "pi_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 1000]),
                    2: np.array([1, 2000, 3, 4]),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 4]),
                    2: np.array([2, 3, 4, 5]),
                },
                "loss_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 4000]),
                    2: np.array([2000, 3, 4, 5]),
                },
                "avg_loss_hessian": np.ones((4, 4)) * 0.5,
            },
            3: {
                "pi_gradients_by_user_id": {
                    1: np.array([1, 2, 3000, 4]),
                    2: np.array([1, 2, 3, 4000]),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([3000, 4, 5, 6]),
                    2: np.array([4, 5, 6, 7000]),
                },
                "loss_gradients_by_user_id": {
                    1: np.array([2, 3, 4, 5]),
                    2: np.array([3, 4, 5, 6]),
                },
                "avg_loss_hessian": np.ones((4, 4)),
            },
        }
        self.sigmoid_1.construct_upper_left_bread_matrix()
        np.testing.assert_equal(
            self.sigmoid_1.upper_left_bread_matrix,
            # Note that this was constructed manually by inserting the correct
            # diagonal blocks and then averaging outer products of the
            # appropriate things in the above algorithm statistics dict
            np.array(
                [
                    [00.5, 00.5, 00.5, 00.5, 0.0, 0.0, 0.0, 0.0],
                    [00.5, 00.5, 00.5, 00.5, 0.0, 0.0, 0.0, 0.0],
                    [00.5, 00.5, 00.5, 00.5, 0.0, 0.0, 0.0, 0.0],
                    [00.5, 00.5, 00.5, 00.5, 0.0, 0.0, 0.0, 0.0],
                    [04.0, 06.5, 09.0, 11.5, 1.0, 1.0, 1.0, 1.0],
                    [05.5, 09.0, 12.5, 16.0, 1.0, 1.0, 1.0, 1.0],
                    [07.0, 11.5, 16.0, 20.5, 1.0, 1.0, 1.0, 1.0],
                    [08.5, 14.0, 19.5, 25.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ),
        )

    def test_calculate_loss_derivatives(self):
        pass

    # Maybe:
    def test_get_loss(self):
        pass

    def test_get_action_prob_pure(self):
        pass

    def test_get_radon_nikodym_weight(self):
        pass

    def test_update_alg(self):
        pass
