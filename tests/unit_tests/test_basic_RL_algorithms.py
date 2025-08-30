import argparse

import pandas as pd
import numpy as np
import jax.numpy as jnp

import basic_RL_algorithms
import constants


class TestSigmoidLS_T3_n2:
    def setup_method(self):
        """
        Runs anew before each test
        """
        self.state_feats = [
            constants.RLStudyArgs.INTERCEPT,
            constants.RLStudyArgs.PAST_REWARD,
        ]
        self.treat_feats = self.state_feats

        args_2 = argparse.Namespace(
            dataset_type="synthetic",
            verbose=0,
            synthetic_mode="delayed_1_dosage",
            RL_alg=constants.RLStudyArgs.SIGMOID_LS_HARD_CLIP,
            N=5,
            n=2,
            upper_clip=0.9,
            lower_clip=0.1,
            fixed_action_prob=0.5,
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
            noise_var=1,
        )

        self.sigmoid_3 = basic_RL_algorithms.SigmoidLS(
            state_feats=self.state_feats,
            treat_feats=self.treat_feats,
            alg_seed=1,
            steepness=args_2.steepness,
            lower_clip=args_2.lower_clip,
            upper_clip=args_2.upper_clip,
            action_centering=args_2.action_centering,
            smooth_clip=(args_2.RL_alg == constants.RLStudyArgs.SIGMOID_LS_SMOOTH_CLIP),
        )

        self.sigmoid_3.all_policies.append(
            {
                "beta_est": jnp.array(
                    [
                        -0.16610159,
                        0.98683333,
                        -1.287509,
                        -1.0602505,
                    ]
                ),
                "seen_user_id": {1, 2},
                "XX": np.ones(self.sigmoid_3.beta_dim),
            }
        )

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
            basic_RL_algorithms.get_pis_batched_sigmoid(
                beta_est=beta_est,
                lower_clip=0.1,
                steepness=10,
                upper_clip=0.9,
                smooth_clip=False,
                batched_treat_states_tensor=treat_states,
            ),
            np.array([0.1693274, 0.1], dtype=np.float32),
        )
