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
            observed_analysis_dict["theta_est"],
            expected_analysis_dict["theta_est"],
            rtol=1e-6,
        )

        # Too hard to go back in time and add expected values for all the keys here,
        # but we can at least check they're present in the observed dict now,
        # and then still compare the most important ones to observed.
        expected_debug_keys = [
            "theta_est",
            "adaptive_sandwich_var_estimate",
            "classical_sandwich_var_estimate",
            "joint_bread_inverse_matrix",
            "joint_bread_matrix",
            "joint_meat_matrix",
            "classical_bread_inverse_matrix",
            "classical_bread_matrix",
            "classical_meat_matrix",
            "all_estimating_function_stacks",
        ]
        assert list(observed_debug_pieces_dict.keys()) == expected_debug_keys

        ### Check joint meat and bread inverse, uniting RL and inference
        np.testing.assert_allclose(
            observed_debug_pieces_dict["joint_meat_matrix"],
            expected_debug_pieces_dict["joint_meat_matrix"],
            atol=1e-5,
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
            atol=1e-8,
        )
        np.testing.assert_allclose(
            observed_analysis_dict["classical_sandwich_var_estimate"],
            expected_analysis_dict["classical_sandwich_var_estimate"],
            atol=1e-8,
        )
