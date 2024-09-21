import numpy as np

import functions_to_pass_to_analysis


def test_get_action_1_prob_pure_no_clip():
    treat_states = np.array([1.0, -1.06434164])
    beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

    np.testing.assert_equal(
        functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure(
            beta_est=beta_est,
            lower_clip=0.1,
            steepness=10,
            upper_clip=0.9,
            treat_states=treat_states,
        ),
        np.array(0.1693274, dtype=np.float32),
    )


def test_get_action_1_prob_pure_clip():
    treat_states = np.array([1.0, -0.12627351])
    beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

    np.testing.assert_equal(
        functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure(
            beta_est=beta_est,
            lower_clip=0.1,
            steepness=10,
            upper_clip=0.9,
            treat_states=treat_states,
        ),
        np.array(0.1, dtype=np.float32),
    )
