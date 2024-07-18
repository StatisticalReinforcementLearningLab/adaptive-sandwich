import argparse

import pandas as pd
import numpy as np
import pytest

import basic_RL_algorithms
import constants


class TestSigmoidLS_T3_n2:
    def setup_method(self):
        """
        Runs anew before each test
        """
        args_1 = argparse.Namespace(
            dataset_type="synthetic",
            verbose=0,
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
        self.state_feats = [
            constants.RLStudyArgs.INTERCEPT,
            constants.RLStudyArgs.PAST_REWARD,
        ]
        self.treat_feats = self.state_feats
        self.sigmoid_1 = basic_RL_algorithms.SigmoidLS(
            state_feats=self.state_feats,
            treat_feats=self.treat_feats,
            alg_seed=1,
            steepness=args_1.steepness,
            lower_clip=args_1.lower_clip,
            upper_clip=args_1.upper_clip,
            action_centering=args_1.action_centering,
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
                "XX": np.ones(self.sigmoid_1.beta_dim),
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
                "XX": np.ones(self.sigmoid_1.beta_dim),
            }
        )

        args_2 = argparse.Namespace(
            dataset_type="synthetic",
            verbose=0,
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
            steepness=10.0,
            alg_state_feats="intercept,past_reward",
            action_centering=1,
            prior="naive",
            T=3,
            recruit_n=50,
            recruit_t=1,
            allocation_sigma=1,
            noise_var=1,
        )

        self.sigmoid_2 = basic_RL_algorithms.SigmoidLS(
            state_feats=self.state_feats,
            treat_feats=self.treat_feats,
            alg_seed=1,
            steepness=args_2.steepness,
            lower_clip=args_2.lower_clip,
            upper_clip=args_2.upper_clip,
            action_centering=args_2.action_centering,
        )

        self.sigmoid_2.all_policies.append(
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
                "XX": np.ones(self.sigmoid_2.beta_dim),
            }
        )

        self.sigmoid_3 = basic_RL_algorithms.SigmoidLS(
            state_feats=self.state_feats,
            treat_feats=self.treat_feats,
            alg_seed=1,
            steepness=args_2.steepness,
            lower_clip=args_2.lower_clip,
            upper_clip=args_2.upper_clip,
            action_centering=args_2.action_centering,
        )

        self.sigmoid_3.all_policies.append(
            {
                "beta_est": pd.DataFrame(
                    {
                        "intercept": [-0.16610159],
                        "past_reward": [0.98683333],
                        "action:intercept": [-1.287509],
                        "action:past_reward": [-1.0602505],
                    }
                ),
                "seen_user_id": {1, 2},
                "XX": np.ones(self.sigmoid_2.beta_dim),
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
                "in_study": [1, 1, 1, 1, 1, 1],
                "action1prob": [0.5, 0.6, 0.7, 0.1, 0.2, 0.3],
            }
        )

        # Time 4 is like time 3 in study df 1 for user 2,
        # whereas user 1 has exited.
        # We use this for an incremental call at time 4.
        self.study_df_5 = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "calendar_t": [1, 2, 3, 4, 1, 2, 3, 4],
                "action": [0, 1, 1, None, None, 1, 1, 0],
                "reward": [1.0, -1, 0, None, None, 1, 1, 1],
                "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1],
                "past_reward": [0.0, 1, -1, None, None, 0, 1, 0],
                "in_study": [1, 1, 1, 0, 0, 1, 1, 1],
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
                "in_study": [1, 1, 1, 1, 1, 1],
            }
        )

        self.study_df_2_incremental = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "calendar_t": [1, 2, 3, 1, 2, 3],
                "action": [0, 1, 0, 1, 1, 0],
                "reward": [1.0, -1, 0, 1, 0, 1],
                "intercept": [1.0, 1, 1, 1, 1, 1],
                "past_reward": [0.0, 1, -1, 1, 1, -10],
                "in_study": [1, 1, 1, 1, 1, 1],
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
                "in_study": [1, 1, 1, 1],
            }
        )

        # Time 1 is like time 2 in study df for user 1,
        # whereas user 2 has not entered yet.
        # We use this for an incremental call at time 1.
        self.study_df_4 = pd.DataFrame(
            {
                "user_id": [
                    1,
                    2,
                ],
                "calendar_t": [
                    1,
                    1,
                ],
                "action": [
                    1,
                    None,
                ],
                "reward": [
                    0,
                    None,
                ],
                "intercept": [1, 1],
                "past_reward": [-1, None],
                "in_study": [1, 0],
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
        self.sigmoid_1.calculate_pi_and_weight_gradients(
            self.study_df_3, 2, self.sigmoid_1.get_current_beta_estimate()
        )
        self.sigmoid_1.calculate_pi_and_weight_gradients(
            self.study_df_1, 3, self.sigmoid_1.get_current_beta_estimate()
        )
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
        self.sigmoid_1.calculate_pi_and_weight_gradients(
            self.study_df_2, 3, self.sigmoid_1.get_current_beta_estimate()
        )
        np.testing.assert_equal(
            self.sigmoid_1.algorithm_statistics_by_calendar_t,
            {
                3: {
                    # Derived by setting a breakpoint in calculate_pi_and_weight_gradients and calling
                    # self.get_action_1_prob_pure(curr_beta_est, self.args.lower_clip, self.args.upper_clip, self.args.steepness, self.get_user_states(current_data, user_id)["treat_states" ][-1])
                    # for each user, then plugging into explicit formula.
                    # prob is 0.26894143 for user 1, .9 for user 2
                    # NOT negative of previous test despite zero action
                    "pi_gradients_by_user_id": {
                        1: np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
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

    def test_calculate_pi_and_weight_gradients_incremental_recruitment(self):
        """
        At time 3, User 1 takes a positive action, meaning positive gradient case, and User 2
        gets clipped at .9, meaning zero gradient.

        In addition, a study df derived from the time 3 study df is passed in for time
        2 (the df doesn't build incrementally from time 2 to 3 but that's fine) simply to test
        that each of the results goes to the correct key.
        """
        self.sigmoid_1.calculate_pi_and_weight_gradients(
            self.study_df_4, 1, self.sigmoid_1.get_current_beta_estimate()
        )
        self.sigmoid_1.calculate_pi_and_weight_gradients(
            self.study_df_5, 4, self.sigmoid_1.get_current_beta_estimate()
        )
        np.testing.assert_equal(
            self.sigmoid_1.algorithm_statistics_by_calendar_t,
            {
                1: {
                    # User 2 not in study
                    "pi_gradients_by_user_id": {
                        1: np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                    "weight_gradients_by_user_id": {
                        1: np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                },
                4: {
                    # User 2 is clipped, User 1 not in study, so all zeros
                    "pi_gradients_by_user_id": {
                        1: np.array([0, 0, 0, 0], dtype="float32"),
                        # Note that these are all zeros because this probability is clipped
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                    "weight_gradients_by_user_id": {
                        1: np.array([0, 0, 0, 0], dtype="float32"),
                        2: np.array([0, 0, 0, 0], dtype="float32"),
                    },
                },
            },
        )

    def test_construct_upper_left_bread_inverse_update_every_decision_no_action_probs_in_loss(
        self,
    ):
        self.sigmoid_1.algorithm_statistics_by_calendar_t = {
            2: {
                "pi_gradients_by_user_id": {
                    1: np.ones(4, dtype="float32"),
                    2: np.ones(4, dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.zeros((4, 1)),
                    2: np.zeros((4, 1)),
                },
            },
            3: {
                "pi_gradients_by_user_id": {
                    1: np.ones(4, dtype="float32"),
                    2: np.ones(4, dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.zeros((4, 2)),
                    2: np.zeros((4, 2)),
                },
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

    def test_construct_upper_left_bread_inverse_update_every_decision_action_probs_in_loss(
        self,
    ):
        self.sigmoid_1.algorithm_statistics_by_calendar_t = {
            2: {
                "pi_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 4], dtype="float32"),
                    2: np.array([3, 4, 5, 6], dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.array(
                        [
                            [1],
                            [3],
                            [5],
                            [7],
                        ],
                        dtype="float32",
                    ),
                    2: np.array(
                        [
                            [1],
                            [4],
                            [6],
                            [8],
                        ],
                        dtype="float32",
                    ),
                },
            },
            3: {
                "pi_gradients_by_user_id": {
                    1: np.array([1, 2, 3, 4], dtype="float32"),
                    2: np.array([3, 4, 5, 6], dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.array(
                        [
                            [1, 2],
                            [3, 4],
                            [5, 6],
                            [7, 8],
                        ],
                        dtype="float32",
                    ),
                    2: np.array(
                        [
                            [1, 1],
                            [3, 4],
                            [5, 6],
                            [7, 8],
                        ],
                        dtype="float32",
                    ),
                },
            },
        }

        # The new contribution to the the bottom left relative to previoius test
        # due to loss_gradient_pi_derivatives_by_user_id being nonzero
        # is average of np.array([[2, 4, 6,  8],
        #                         [4, 8, 12, 16],
        #                         [6, 12, 18, 24],
        #                         [8, 16, 24, 32]])
        # and np.array([[3, 4, 5, 6],
        #               [12, 16, 20, 24],
        #               [18, 24, 30, 36],
        #               [24, 32, 40, 48]])
        # which is
        #     np.array([[ 2.5,  4. ,  5.5,  7. ],
        #               [ 8. , 12. , 16. , 20. ],
        #               [ 12. , 18. , 24. , 30. ],
        #               [ 16. , 24. , 32. , 40. ]])
        #

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
                    [06.5, 10.5, 14.5, 18.5, 1.0, 1.0, 1.0, 1.0],
                    [13.5, 21.0, 28.5, 36.0, 1.0, 1.0, 1.0, 1.0],
                    [19.0, 29.5, 40.0, 50.5, 1.0, 1.0, 1.0, 1.0],
                    [24.5, 38.0, 51.5, 65.0, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype="float32",
            ),
        )

    def test_construct_upper_left_bread_inverse_2_decs_btwn_updates_no_action_probs_in_loss(
        self,
    ):
        self.sigmoid_1.algorithm_statistics_by_calendar_t = {
            3: {
                "pi_gradients_by_user_id": {
                    1: np.ones(4, dtype="float32"),
                    2: np.ones(4, dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.zeros((4, 2)),
                    2: np.zeros((4, 2)),
                },
            },
            4: {
                "pi_gradients_by_user_id": {
                    1: np.ones(4, dtype="float32"),
                    2: np.ones(4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([3, 4, 5, 6], dtype="float32"),
                    2: np.array([4, 5, 6, 7], dtype="float32"),
                },
            },
            5: {
                "pi_gradients_by_user_id": {
                    1: np.ones(4, dtype="float32"),
                    2: np.ones(4, dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.zeros((4, 4)),
                    2: np.zeros((4, 4)),
                },
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

    def test_construct_upper_left_bread_inverse_2_decs_btwn_updates_action_probs_in_loss(
        self,
    ):
        self.sigmoid_1.algorithm_statistics_by_calendar_t = {
            3: {
                "pi_gradients_by_user_id": {
                    1: np.ones(4, dtype="float32"),
                    2: np.ones(4, dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.array(
                        [
                            [1, 2],
                            [3, 4],
                            [5, 6],
                            [7, 8],
                        ],
                        dtype="float32",
                    ),
                    2: np.array(
                        [
                            [1, 1],
                            [3, 4],
                            [5, 6],
                            [7, 8],
                        ],
                        dtype="float32",
                    ),
                },
            },
            4: {
                "pi_gradients_by_user_id": {
                    1: 2 * np.ones(4, dtype="float32"),
                    2: np.ones(4, dtype="float32"),
                },
                "weight_gradients_by_user_id": {
                    1: np.array([3, 4, 5, 6], dtype="float32"),
                    2: np.array([4, 5, 6, 7], dtype="float32"),
                },
            },
            5: {
                "pi_gradients_by_user_id": {
                    1: np.ones(4, dtype="float32"),
                    2: 2 * np.ones(4, dtype="float32"),
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
                "loss_gradient_pi_derivatives_by_user_id": {
                    1: np.array(
                        [
                            [1, 2, 1, 1],
                            [3, 4, 1, 2],
                            [5, 6, 1, 1],
                            [7, 8, 2, 2],
                        ],
                        dtype="float32",
                    ),
                    2: np.array(
                        [
                            [1, 1, 2, 1],
                            [3, 4, 3, 2],
                            [5, 6, 1, 1],
                            [7, 8, 1, 2],
                        ],
                        dtype="float32",
                    ),
                },
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

        # The new contribution to the the bottom left relative to previous test
        # due to loss_gradient_pi_derivatives_by_user_id being nonzero
        # is average of np.array([[3., 3., 3., 3.],
        #                         [5., 5., 5., 5.],
        #                         [3., 3., 3., 3.],
        #                         [6., 6., 6., 6.]], dtype=float32)
        # and np.array([[3., 3., 3., 3.],
        #               [5., 5., 5., 5.],
        #               [2., 2., 2., 2.],
        #               [3., 3., 3., 3.]], dtype=float32)
        # which is
        #     np.array([[ 3,  3. ,  3,  3. ],
        #               [ 5. , 5. , 5. , 5. ],
        #               [ 2.5 , 2.5 , 2.5 , 2.5 ],
        #               [ 4.5 , 4.5 , 4.5 , 4.5 ]])

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
                    [11.0, 14.0, 17.0, 20.0, 1.0, 1.0, 1.0, 1.0],
                    [18.0, 23.0, 28.0, 33.0, 1.0, 1.0, 1.0, 1.0],
                    [20.5, 27.5, 34.5, 41.5, 1.0, 1.0, 1.0, 1.0],
                    [27.5, 36.5, 45.5, 54.5, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype="float32",
            ),
        )

    @pytest.mark.skip(reason="To be implemented")
    def test_construct_upper_left_bread_inverse_incremental_recruitment(self):
        raise NotImplementedError()

    @pytest.mark.skip(reason="To be implemented")
    def test_construct_upper_left_bread_inverse_two_updates(self):
        raise NotImplementedError()

    def test_calculate_loss_derivatives_no_action_centering(self):
        self.sigmoid_1.calculate_loss_derivatives(
            self.study_df_1, 3, self.sigmoid_1.get_current_beta_estimate()
        )
        np.testing.assert_equal(
            self.sigmoid_1.algorithm_statistics_by_calendar_t,
            {
                # Derived by doing loss calculation by hand and also setting
                # a breakpoint inside get_loss, verifying pieces incrementally.
                # Hessians calculated manually using formula for each user and averaged
                # See test with action centering for a better step-by-step
                # derivation that can be replicated.
                4: {
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
                    "loss_gradient_pi_derivatives_by_user_id": {
                        1: np.zeros((4, 3), dtype="float32"),
                        2: np.zeros((4, 3), dtype="float32"),
                    },
                }
            },
        )

    def test_calculate_loss_derivatives_action_centering(self):
        self.sigmoid_2.calculate_loss_derivatives(
            self.study_df_1, 3, self.sigmoid_2.get_current_beta_estimate()
        )

        beta = np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32")

        user_1_centered_actions = np.array([0 - 0.5, 1 - 0.6, 1 - 0.7], dtype="float32")
        user_1_states = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]], dtype="float32")
        user_1_rewards = np.array([1.0, -1, 0], dtype="float32")
        user_1_loss_gradient = -2 * sum(
            (
                (
                    user_1_rewards[i]
                    - beta[:2] @ user_1_states[i]
                    - beta[2:] @ (user_1_centered_actions[i] * user_1_states[i])
                )
                * np.concatenate(
                    [
                        user_1_states[i],
                        user_1_centered_actions[i] * user_1_states[i],
                    ]
                )
                for i in range(3)
            )
        )

        user_2_centered_actions = np.array([1 - 0.1, 1 - 0.2, 0 - 0.3], dtype="float32")
        user_2_states = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype="float32")
        user_2_rewards = np.array([1.0, 0, 1.0], dtype="float32")
        user_2_loss_gradient = -2 * sum(
            [
                (
                    user_2_rewards[i]
                    - beta[:2] @ user_2_states[i]
                    - beta[2:] @ (user_2_centered_actions[i] * user_2_states[i])
                )
                * np.concatenate(
                    [
                        user_2_states[i],
                        user_2_centered_actions[i] * user_2_states[i],
                    ]
                )
                for i in range(3)
            ]
        )

        # There are small numerical differences between the above calculations
        # and the real results. Assert they are close here and then just use
        # the real results nested in the algorithm stats dict below
        # instead of ironing out floating point issues
        np.testing.assert_allclose(
            user_1_loss_gradient,
            np.array([-4.0000005, 16.199999, 5.3599997, 5.8199997], dtype="float32"),
            atol=1e-05,
        )
        np.testing.assert_allclose(
            user_2_loss_gradient,
            np.array([20.0, 25.8, 23.64, 21.9], dtype="float32"),
            atol=1e-05,
        )

        # This is a little lazy but the loss gradients match up, it suggests action
        # centering is being incorporated correctly into the loss, and I will
        # simply take the hessian and pi derivatives being computed and
        # use them as the expected values because the code is behaving correctly
        # in simulations and the addition of action centering doesn't add
        # further difficulties to the JAX gradient infrastructure.

        np.testing.assert_equal(
            self.sigmoid_2.algorithm_statistics_by_calendar_t,
            {
                4: {
                    "loss_gradients_by_user_id": {
                        1: np.array(
                            [-4.0000005, 16.199999, 5.3599997, 5.8199997],
                            dtype="float32",
                        ),
                        2: np.array([20.0, 25.8, 23.64, 21.9], dtype="float32"),
                    },
                    "avg_loss_hessian": np.array(
                        [
                            [6.0, 2.0, 1.6000001, 1.8],
                            [2.0, 4.0, 1.8, 2.4],
                            [1.6000001, 1.8, 2.04, 1.5199999],
                            [1.8, 2.4, 1.5199999, 1.6999999],
                        ],
                        dtype="float32",
                    ),
                    "loss_gradient_pi_derivatives_by_user_id": {
                        1: np.array(
                            [
                                [-6.0, -14.0, 2.0],
                                [-0.0, -14.0, -2.0],
                                [10.0, -15.199999, 7.2],
                                [-0.0, -15.199999, -7.2],
                            ],
                            dtype="float32",
                        ),
                        2: np.array(
                            [
                                [-14.0, -14.0, -6.0],
                                [-14.0, -14.0, -0.0],
                                [-25.2, -24.4, 7.6000004],
                                [-25.199999, -24.400002, -0.0],
                            ],
                            dtype="float32",
                        ),
                    },
                }
            },
        )

    @pytest.mark.skip(reason="To be implemented")
    def test_calculate_loss_derivatives_incremental_recruitment(self):
        raise NotImplementedError()

    # TODO: Add test of loss derivatives with multiple updates? Had a case that
    # only broke on multiple updates...

    # Indirectly tested by testing its gradient is well-formed... but could add something direct.
    def test_get_radon_nikodym_weight(self):
        pass

    def test_get_action_probs(self):
        curr_timestep_data = pd.DataFrame(
            {
                "user_id": [
                    1,
                    2,
                ],
                "calendar_t": [1, 1],
                "action": [
                    0,
                    1,
                ],
                "reward": [1.0, 1],
                "intercept": [1.0, 1],
                "past_reward": [-1.06434164, -0.12627351],
            }
        )
        np.testing.assert_equal(
            self.sigmoid_3.get_action_probs(curr_timestep_data),
            np.array([0.1693274, 0.1], dtype=np.float32),
        )

    def test_update_alg(self):
        pass

    def test_get_pis_batched(self):
        treat_states = np.array([[1.0, -1.06434164], [1.0, -0.12627351]])
        beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

        np.testing.assert_equal(
            basic_RL_algorithms.get_pis_batched(
                beta_est=beta_est,
                lower_clip=0.1,
                steepness=10,
                upper_clip=0.9,
                batched_treat_states_tensor=treat_states,
            ),
            np.array([0.1693274, 0.1], dtype=np.float32),
        )

    def test_get_action_1_prob_pure_no_clip(self):
        treat_states = np.array([1.0, -1.06434164])
        beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

        np.testing.assert_equal(
            basic_RL_algorithms.get_action_1_prob_pure(
                beta_est=beta_est,
                lower_clip=0.1,
                steepness=10,
                upper_clip=0.9,
                treat_states=treat_states,
            ),
            np.array(0.1693274, dtype=np.float32),
        )

    def test_get_action_1_prob_pure_clip(self):
        treat_states = np.array([1.0, -0.12627351])
        beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

        np.testing.assert_equal(
            basic_RL_algorithms.get_action_1_prob_pure(
                beta_est=beta_est,
                lower_clip=0.1,
                steepness=10,
                upper_clip=0.9,
                treat_states=treat_states,
            ),
            np.array(0.1, dtype=np.float32),
        )
