import argparse

import pandas as pd
import numpy as np

import basic_RL_algorithms
import constants


class TestSigmoidLS_T3_n2:
    def setup_method(self):
        """
        Runs anew before each test
        """
        self.args_1 = argparse.Namespace(
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
        self.state_feats_1 = [
            constants.RLStudyArgs.INTERCEPT,
            constants.RLStudyArgs.PAST_REWARD,
        ]
        self.treat_feats_1 = self.state_feats_1
        self.sigmoid_1 = basic_RL_algorithms.SigmoidLS(
            self.args_1,
            self.state_feats_1,
            self.treat_feats_1,
            alg_seed=1,
            allocation_sigma=self.args_1.allocation_sigma,
            steepness=self.args_1.steepness,
        )
        self.sigmoid_1.all_policies.append(
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
        self.sigmoid_1.all_policies.append(
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

        self.study_df_1 = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "calendar_t": [1, 2, 3, 1, 2, 3],
                "action": [0, 1, 1, 1, 1, 0],
                "reward": [1.0, -1, 0, 1, 0, 1],
                "intercept": [1.0, 1, 1, 1, 1, 1],
                "past_reward": [0.0, 1, -1, 1, 1, 0],
            }
        )

        self.study_df_2 = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "calendar_t": [1, 2, 3, 1, 2, 3],
                "action": [0, 1, 0, 1, 1, 0],
                "reward": [1.0, -1, 0, 1, 0, 1],
                "intercept": [1.0, 1, 1, 1, 1, 1],
                "past_reward": [0.0, 1, -1, 1, 1, -10],
            }
        )

        # Both users like study_df_1 user 1 final two time steps
        self.study_df_3 = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "calendar_t": [1, 2, 1, 2],
                "action": [1, 1, 1, 1],
                "reward": [-1, 0, -1, 0],
                "intercept": [1, 1, 1, 1],
                "past_reward": [1, -1, 1, -1],
            }
        )

    def test_calculate_pi_and_weight_gradients_positive_action_high_clip(self):
        """
        At time 3, User 1 takes a positive action, meaning positive gradient case, and User 2
        gets clipped at .9, meaning zero gradient.

        In addition, a study df derived from the time 3 study df is passed in for time
        2 (the df doesn't build incrementally from time 2 to 3 but that's fine) simply to test
        that each of the results goes to the correct key.
        """
        self.sigmoid_1.calculate_pi_and_weight_gradients(self.study_df_3, 2)
        self.sigmoid_1.calculate_pi_and_weight_gradients(self.study_df_1, 3)
        np.testing.assert_equal(
            self.sigmoid_1.algorithm_statistics_by_calendar_t,
            {
                2: {
                    # Derived by slightly tweaking the df for time 3 to get manually
                    # predictable results.
                    "pi_gradients_by_user_id": {
                        1: np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                        2: np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                    },
                    "weight_gradients_by_user_id": {
                        1: np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                        2: np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                    },
                },
                3: {
                    # derived by setting a breakpoint and calling
                    # self.get_action_prob_pure(curr_beta_est, self.args.lower_clip, self.args.upper_clip,
                    #                           self.get_user_states(current_data, 1)["treat_states" ][-1])
                    # for each user, then plugging into explicit formula. 0.26894143 for user 1, .9 for user 2
                    "pi_gradients_by_user_id": {
                        1: np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                        # Note that these are all zeros because this probability is clipped
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                    # derived using pi and pi gradients from above (two derivative cases depending
                    # on action are easy to calculate)
                    "weight_gradients_by_user_id": {
                        1: np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                },
            },
        )

    def test_calculate_pi_and_weight_gradients_zero_action_low_clip(self):
        """
        User 1 takes no action, meaning negative gradient case, and User 2
        gets clipped at .1, meaning zero gradient.
        """
        self.sigmoid_1.calculate_pi_and_weight_gradients(self.study_df_2, 3)
        np.testing.assert_equal(
            self.sigmoid_1.algorithm_statistics_by_calendar_t,
            {
                3: {
                    # Derived by setting a breakpoint in calculate_pi_and_weight_gradients and calling
                    # self.get_action_prob_pure(curr_beta_est, self.args.lower_clip, self.args.upper_clip, self.get_user_states(current_data, user_id)["treat_states" ][-1])
                    # for each user, then plugging into explicit formula.
                    # prob is 0.26894143 for user 1, .9 for user 2
                    # Negative of previous test because zero action
                    "pi_gradients_by_user_id": {
                        1: np.array([0, 0, -0.19661194, 0.19661194], dtype="float32"),
                        # Note that these are all zeros because this probability is clipped
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                    # derived using pi and pi gradients from above (two derivative cases depending
                    # on action are easy to calculate)
                    "weight_gradients_by_user_id": {
                        1: np.array([0, 0, -0.26894143, 0.26894143], dtype="float32"),
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                },
            },
        )

    # TODO: Test extra hessian term for action centering
    def test_construct_upper_left_bread_inverse_update_every_decision(self):
        self.sigmoid_1.algorithm_statistics_by_calendar_t = {
            2: {
                "pi_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 4], dtype="float32"),
                    2: np.array([2, 3, 4, 5], dtype="float32"),
                },
                "loss_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "avg_loss_hessian": np.ones((4, 4)) * -1,
            },
            3: {
                "pi_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "loss_gradients_by_user_id": {
                    1: np.array([2, 3, 4, 5], dtype="float32"),
                    2: np.array([3, 4, 5, 6], dtype="float32"),
                },
                "avg_loss_hessian": np.ones((4, 4)),
            },
        }
        self.sigmoid_1.construct_upper_left_bread_inverse()
        np.testing.assert_equal(
            self.sigmoid_1.upper_left_bread_inverse,
            # Note that this was constructed manually by inserting the correct
            # diagonal blocks and then averaging outer products of the
            # appropriate things in the above algorithm statistics dict
            np.array(
                [
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [04.0, 06.5, 09.0, 11.5, 1.0, 1.0, 1.0, 1.0],
                    [05.5, 09.0, 12.5, 16.0, 1.0, 1.0, 1.0, 1.0],
                    [07.0, 11.5, 16.0, 20.5, 1.0, 1.0, 1.0, 1.0],
                    [08.5, 14.0, 19.5, 25.0, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype="float32",
            ),
        )

    def test_construct_upper_left_bread_inverse_2_decs_btwn_updates(self):
        self.sigmoid_1.algorithm_statistics_by_calendar_t = {
            3: {
                "pi_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 4], dtype="float32"),
                    2: np.array([2, 3, 4, 5], dtype="float32"),
                },
                "loss_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "avg_loss_hessian": np.ones((4, 4)) * -1,
            },
            4: {
                "pi_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([3, 4, 5, 6], dtype="float32"),
                    2: np.array([4, 5, 6, 7], dtype="float32"),
                },
            },
            5: {
                "pi_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "loss_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 4], dtype="float32"),
                    2: np.array([2, 3, 4, 5], dtype="float32"),
                },
                "avg_loss_hessian": np.ones((4, 4)) * 1,
            },
            6: {
                "pi_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([None] * 4, dtype="float32"),
                    2: np.array([None] * 4, dtype="float32"),
                },
            },
        }
        self.sigmoid_1.construct_upper_left_bread_inverse()
        np.testing.assert_equal(
            self.sigmoid_1.upper_left_bread_inverse,
            # Note that this was constructed manually by inserting the correct
            # diagonal blocks and then summing weight gradients and averaging outer
            # products of appropriate things in the above algorithm statistics dict
            np.array(
                [
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [08.0, 11.0, 14.0, 17.0, 1.0, 1.0, 1.0, 1.0],
                    [13.0, 18.0, 23.0, 28.0, 1.0, 1.0, 1.0, 1.0],
                    [18.0, 25.0, 32.0, 39.0, 1.0, 1.0, 1.0, 1.0],
                    [23.0, 32.0, 41.0, 50.0, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype="float32",
            ),
        )

    def test_calculate_loss_derivatives(self):
        self.sigmoid_1.calculate_loss_derivatives(self.study_df_1, 2)
        np.testing.assert_equal(
            self.sigmoid_1.algorithm_statistics_by_calendar_t,
            {
                # Derived by doing loss calculation by hand and also setting
                # a breakpoint inside get_loss, verifying pieces incrementally.
                # Hessians calculated manually using formula for each user and averaged
                3: {
                    "loss_gradients_by_user_id": {
                        1: np.array([6, 26, 10, 26], dtype="float32"),
                        2: np.array([26, 30, 30, 30], dtype="float32"),
                    },
                    "avg_loss_hessian": np.array(
                        [
                            [6, 2, 4, 2],
                            [2, 4, 2, 4],
                            [4, 2, 4, 2],
                            [2, 4, 2, 4],
                        ],
                        dtype="float32",
                    ),
                }
            },
        )

    # TODO: Add test of loss derivatives with multiple updates? Had a case that
    # only broke on multiple updates...

    # TODO: Should integrate next two functions with actual algorithm logic before testing
    # UPDATE: No I shouldn't. Don't need to put that effort in for sample
    def test_get_loss(self):
        pass

    def test_get_action_prob_pure(self):
        pass

    # Indirectly tested by testing its gradient is well-formed
    def test_get_radon_nikodym_weight(self):
        pass

    def test_update_alg(self):
        pass
