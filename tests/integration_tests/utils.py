import os
import pickle

import numpy as np
import pandas as pd


def get_abs_path(code_path, relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(code_path), relative_path))


def assert_real_run_output_as_expected(test_file_path, relative_path_to_output_dir):
    # Load the observed and expected pickle files
    with open(
        get_abs_path(
            test_file_path,
            f"{relative_path_to_output_dir}/study_df.pkl",
        ),
        "rb",
    ) as observed_study_df_pickle, open(
        get_abs_path(
            test_file_path,
            f"{relative_path_to_output_dir}/analysis.pkl",
        ),
        "rb",
    ) as observed_analysis_pickle, open(
        get_abs_path(
            test_file_path,
            f"{relative_path_to_output_dir}/debug_pieces.pkl",
        ),
        "rb",
    ) as observed_debug_pieces_pickle, open(
        get_abs_path(test_file_path, "expected_study_df.pkl"),
        "rb",
    ) as expected_study_df_pickle, open(
        get_abs_path(test_file_path, "expected_analysis.pkl"),
        "rb",
    ) as expected_analysis_pickle, open(
        get_abs_path(test_file_path, "expected_debug_pieces.pkl"),
        "rb",
    ) as expected_debug_pieces_pickle:
        observed_study_df = pickle.load(observed_study_df_pickle)
        observed_analysis_dict = pickle.load(observed_analysis_pickle)
        observed_debug_pieces_dict = pickle.load(observed_debug_pieces_pickle)

        # The expected df is generated from a time when these were set as Int64.
        # I don't remember why we had to change to float64; I believe it was
        # necessary on the analysis side.
        expected_study_df = pickle.load(expected_study_df_pickle).astype(
            {"policy_num": "float64", "action": "float64"}
        )
        expected_analysis_dict = pickle.load(expected_analysis_pickle)
        expected_debug_pieces_dict = pickle.load(expected_debug_pieces_pickle)

        ### Check base study dataframes equal. This is important so that
        # we are even in the game, trying to produce the right inference results.
        pd.testing.assert_frame_equal(observed_study_df, expected_study_df)

        # Check that we have the same theta estimate in both cases.
        np.testing.assert_allclose(
            observed_analysis_dict["theta_est"], expected_analysis_dict["theta_est"]
        )

        assert observed_debug_pieces_dict.keys() == expected_debug_pieces_dict.keys()

        ### Check RL-side derivatives and bread contribution

        # Note this may start after the first update.  If the observed starts
        # before that, we don't care.
        for t in expected_debug_pieces_dict[
            "algorithm_statistics_by_calendar_t"
        ].keys():
            for k in [
                "loss_gradients_by_user_id",
                "loss_gradient_pi_derivatives_by_user_id",
                "pi_gradients_by_user_id",
                "weight_gradients_by_user_id",
            ]:
                # We only expect these keys at the first time after each update
                if (
                    k
                    in (
                        "loss_gradients_by_user_id",
                        "loss_gradient_pi_derivatives_by_user_id",
                    )
                    and k
                    not in expected_debug_pieces_dict[
                        "algorithm_statistics_by_calendar_t"
                    ][t]
                ):
                    continue

                # Check all the same users present for this key
                assert (
                    observed_debug_pieces_dict["algorithm_statistics_by_calendar_t"][t][
                        k
                    ].keys()
                    == expected_debug_pieces_dict["algorithm_statistics_by_calendar_t"][
                        t
                    ][k].keys()
                ), f"Keys don't match for t={t} and k={k}"

                # Now compare the values for each user
                for user_id in expected_debug_pieces_dict[
                    "algorithm_statistics_by_calendar_t"
                ][t][k].keys():
                    np.testing.assert_allclose(
                        observed_debug_pieces_dict[
                            "algorithm_statistics_by_calendar_t"
                        ][t][k][user_id],
                        expected_debug_pieces_dict[
                            "algorithm_statistics_by_calendar_t"
                        ][t][k][user_id],
                        err_msg=f"Mismatch for t={t}, k={k}, user_id={user_id}",
                    )

            # We only expect this key at the first time after each update
            if (
                "avg_loss_hessian"
                not in expected_debug_pieces_dict["algorithm_statistics_by_calendar_t"][
                    t
                ]
            ):
                continue
            np.testing.assert_allclose(
                observed_debug_pieces_dict["algorithm_statistics_by_calendar_t"][t][
                    "avg_loss_hessian"
                ],
                expected_debug_pieces_dict["algorithm_statistics_by_calendar_t"][t][
                    "avg_loss_hessian"
                ],
                atol=1e-5,
                err_msg=f"Mismatch for t={t}, k=avg_loss_hessian",
            )

        np.testing.assert_allclose(
            observed_debug_pieces_dict["upper_left_bread_inverse"],
            expected_debug_pieces_dict["upper_left_bread_inverse"],
            atol=1e-5,
        )

        ### Check inference-side derivatives

        np.testing.assert_allclose(
            observed_debug_pieces_dict["inference_loss_gradients"],
            expected_debug_pieces_dict["inference_loss_gradients"],
        )

        np.testing.assert_allclose(
            observed_debug_pieces_dict["inference_loss_hessians"],
            expected_debug_pieces_dict["inference_loss_hessians"],
            atol=1e-5,
        )

        np.testing.assert_allclose(
            observed_debug_pieces_dict["inference_loss_gradient_pi_derivatives"],
            expected_debug_pieces_dict["inference_loss_gradient_pi_derivatives"],
        )

        ### Check joint meat and bread inverse, uniting RL and inference
        np.testing.assert_allclose(
            observed_debug_pieces_dict["joint_meat_matrix"],
            expected_debug_pieces_dict["joint_meat_matrix"],
        )

        np.testing.assert_allclose(
            observed_debug_pieces_dict["joint_bread_inverse_matrix"],
            expected_debug_pieces_dict["joint_bread_inverse_matrix"],
            atol=1e-5,
        )

        ### Check final results
        np.testing.assert_allclose(
            observed_analysis_dict["adaptive_sandwich_var_estimate"],
            expected_analysis_dict["adaptive_sandwich_var_estimate"],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            observed_analysis_dict["classical_sandwich_var_estimate"],
            expected_analysis_dict["classical_sandwich_var_estimate"],
            atol=1e-9,
        )
