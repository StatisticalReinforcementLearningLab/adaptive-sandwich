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
            action_centering=0,
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
        )

        self.sigmoid_2.all_policies.append(
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
                "XX": np.ones(self.sigmoid_1.beta_dim),
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
            }
        )

        self.study_df_1_incremental = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
                "calendar_t": [1, 2, 3, 4, 1, 2, 3, 4],
                "action": [0, 1, 1, None, None, 1, 1, 0],
                "reward": [1.0, -1, 0, None, None, 1, 0, 1],
                "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1],
                "past_reward": [0.0, 1, -1, None, None, 1, 1, 0],
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

        self.study_df_3_incremental = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "calendar_t": [1, 2, 3, 1, 2, 3],
                "action": [1, 1, None, None, 1, 1],
                "reward": [-1, 0, None, None, -1, 0],
                "intercept": [1, 1, 1, 1, 1, 1],
                "past_reward": [1, -1, None, None, 1, -1],
                "in_study": [1, 1, 0, 0, 1, 1],
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
        self.sigmoid_1.calculate_loss_derivatives(
            self.study_df_1, 2, self.sigmoid_1.get_current_beta_estimate()
        )
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
            self.sigmoid_2.get_action_probs(curr_timestep_data),
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
