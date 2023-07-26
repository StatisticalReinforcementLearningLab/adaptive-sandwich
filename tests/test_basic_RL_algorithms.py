import argparse

import basic_RL_algorithms
import constants


class TestSigmoidLS:
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
    state_feats_1 = [
        constants.RLStudyArgs.INTERCEPT,
        constants.RLStudyArgs.TIME_OF_DAY,
        constants.RLStudyArgs.PRIOR_DAY_BRUSH,
    ]
    treat_feats_1 = state_feats_1
    sigmoid_1 = basic_RL_algorithms.SigmoidLS(
        args_1,
        state_feats_1,
        treat_feats_1,
        alg_seed=1,
        allocation_sigma=args_1.allocation_sigma,
        steepness=args_1.steepness,
    )

    def test_calculate_pi_and_weight_gradients(self):
        pass

    def test_construct_upper_left_bread_matrix(self):
        pass

    def test_calculate_loss_derivatives(self):
        pass

    def test_update_alg(self):
        pass

    # Maybe:
    def test_get_loss(self):
        pass

    def test_get_action_prob_pure(self):
        pass

    def test_get_radon_nikodym_weight(self):
        pass
