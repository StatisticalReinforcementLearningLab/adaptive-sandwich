import glob
import logging
import pickle
import sys
import os

import click
import numpy as np
import pandas as pd
import scipy.stats
import plotext as plt
import seaborn as sns
import matplotlib.pyplot as pyplt

from .helper_functions import get_action_1_fraction, get_action_prob_variance

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


@click.command()
@click.option(
    "--input_glob",
    help="A glob that captures all of the analyses to be collected.  Leaf folders will be searched for analyses",
    required=True,
)
@click.option("--num_users", type=int, required=True)
@click.option(
    "--index_to_check_ci_coverage",
    type=int,
    help="The index of the parameter to check confidence interval coverage for across runs.  If not provided, coverage will not be checked.",
)
@click.option(
    "--in_study_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the study dataframe that indicates whether a user is in the study.",
)
@click.option(
    "--action_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates the action taken by the user.",
)
@click.option(
    "--action_prob_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates the probability of taking action 1.",
)
@click.option(
    "--study_df_filename",
    type=str,
    help="The filename of the pickled study DataFrame.  This is not the full path.",
    required=True,
)
def collect_existing_analyses(
    input_glob: str,
    num_users: int,
    index_to_check_ci_coverage: int,
    in_study_col_name: str,
    action_col_name: str,
    action_prob_col_name: str,
    study_df_filename: str,
) -> None:
    """
    Collects existing analyses from the specified input glob and computes the mean parameter estimate,
    empirical variance, and adaptive/classical sandwich variance estimates.
    Optionally checks confidence interval coverage for a specified parameter index.

    Args:
        input_glob (str): The glob pattern to search for analysis files.
        num_users (int): The number of users in the study.
        index_to_check_ci_coverage (int, optional): The index of the parameter to check confidence
            interval coverage for. If not provided, coverage will not be checked.
        in_study_col_name (str): The name of the column indicating whether a user is in
            the study.
        action_col_name (str): The name of the column indicating the action taken by the
            user.
        action_prob_col_name (str): The name of the column indicating the probability of
            taking action 1.
        study_df_filename (str): The filename of the pickled study DataFrame. Not the full path.
    """

    raw_theta_estimates = []
    raw_adaptive_sandwich_var_estimates = []
    raw_classical_sandwich_var_estimates = []
    filenames = glob.glob(input_glob)

    logger.info("Found %d files under the glob %s", len(filenames), input_glob)
    if len(filenames) == 0:
        raise RuntimeError("Aborting because no files found. Please check path.")

    max_adaptive_estimate_at_index_filename = None
    max_adaptive_estimate_at_index = -np.inf

    # Summary metrics to reduce memory footprint
    condition_numbers = []
    condition_numbers_first_block = []
    action_1_fractions = []
    action_prob_variances = []
    min_eigvals_first_block = []
    max_eigvals_first_block = []
    first_beta_coords = []
    identity_diff_abs_maxes = []
    identity_diff_frobenius_norms = []

    for i, filename in enumerate(filenames):
        if i and len(filenames) >= 10 and i % (len(filenames) // 10) == 0:
            logger.info("A(nother) tenth of files processed.")
        if not os.stat(filename).st_size:
            raise RuntimeError(
                "Empty analysis pickle.  This means there were probably timeouts or other failures during simulations."
            )
        with open(filename, "rb") as f:
            analysis_dict = pickle.load(f)
            (
                theta_est,
                adaptive_sandwich_var,
                classical_sandwich_var,
            ) = (
                analysis_dict["theta_est"],
                analysis_dict["adaptive_sandwich_var_estimate"],
                analysis_dict["classical_sandwich_var_estimate"],
            )
            raw_theta_estimates.append(theta_est)
            raw_adaptive_sandwich_var_estimates.append(adaptive_sandwich_var)
            raw_classical_sandwich_var_estimates.append(classical_sandwich_var)

            if index_to_check_ci_coverage is not None:
                adaptive_estimate_at_index = adaptive_sandwich_var[
                    index_to_check_ci_coverage
                ][index_to_check_ci_coverage]

                if adaptive_estimate_at_index > max_adaptive_estimate_at_index:
                    max_adaptive_estimate_at_index = adaptive_estimate_at_index
                    max_adaptive_estimate_at_index_filename = filename
        # Load and extract summary from debug pieces, then discard full object
        with open(filename.replace("analysis.pkl", "debug_pieces.pkl"), "rb") as f:
            debug_pieces = pickle.load(f)
            if "joint_bread_inverse_condition_number" in debug_pieces:
                condition_numbers.append(
                    debug_pieces["joint_bread_inverse_condition_number"]
                )
            if "joint_bread_inverse_first_block_eigvals" in debug_pieces:
                eigs = debug_pieces["joint_bread_inverse_first_block_eigvals"]
                min_eigvals_first_block.append(np.min(eigs))
                max_eigvals_first_block.append(np.max(eigs))
            if "all_post_update_betas" in debug_pieces:
                first_beta = debug_pieces["all_post_update_betas"][0]
                first_beta_coords.append(first_beta)
            if "identity_diff_abs_max" in debug_pieces:
                identity_diff_abs_maxes.append(debug_pieces["identity_diff_abs_max"])
            if "identity_diff_frobenius_norm" in debug_pieces:
                identity_diff_frobenius_norms.append(
                    debug_pieces["identity_diff_frobenius_norm"]
                )
            if "joint_bread_inverse_first_block_condition_number" in debug_pieces:
                condition_numbers_first_block.append(
                    debug_pieces["joint_bread_inverse_first_block_condition_number"]
                )

            # Discard debug_pieces to free memory
            del debug_pieces

        # Load and extract summary from study dataframe, then discard full object
        with open(filename.replace("analysis.pkl", study_df_filename), "rb") as f:
            study_df = pd.read_pickle(f)
            action_1_fractions.append(
                get_action_1_fraction(study_df, in_study_col_name, action_col_name)
            )
            action_prob_variances.append(
                get_action_prob_variance(
                    study_df, in_study_col_name, action_prob_col_name
                )
            )
            # Discard study_df to free memory
            del study_df

    theta_estimates = np.array(raw_theta_estimates)
    adaptive_sandwich_var_estimates = np.array(raw_adaptive_sandwich_var_estimates)
    classical_sandwich_var_estimates = np.array(raw_classical_sandwich_var_estimates)

    mean_theta_estimate = np.mean(theta_estimates, axis=0)
    empirical_var_normalized = np.atleast_2d(np.cov(theta_estimates.T, ddof=1))

    mean_adaptive_sandwich_var_estimate = np.mean(
        adaptive_sandwich_var_estimates, axis=0
    )
    median_adaptive_sandwich_var_estimate = np.median(
        adaptive_sandwich_var_estimates, axis=0
    )
    adaptive_sandwich_var_estimate_std_deviations = np.sqrt(
        np.var(adaptive_sandwich_var_estimates, axis=0, ddof=1)
    )
    adaptive_sandwich_var_estimate_mins = np.min(
        adaptive_sandwich_var_estimates, axis=0
    )
    adaptive_sandwich_var_estimate_maxes = np.max(
        adaptive_sandwich_var_estimates, axis=0
    )

    mean_classical_sandwich_var_estimate = np.mean(
        classical_sandwich_var_estimates, axis=0
    )
    median_classical_sandwich_var_estimate = np.median(
        classical_sandwich_var_estimates, axis=0
    )
    classical_sandwich_var_estimate_std_deviations = np.sqrt(
        np.var(classical_sandwich_var_estimates, axis=0, ddof=1)
    )
    classical_sandwich_var_estimate_mins = np.min(
        classical_sandwich_var_estimates, axis=0
    )
    classical_sandwich_var_estimate_maxes = np.max(
        classical_sandwich_var_estimates, axis=0
    )

    # Calculate standard error (or corresponding variance) of variance estimate for each
    # component of theta.  This is done by finding an unbiased estimator of the standard
    # formula for the standard error of a variance from iid observations.
    # Population standard error formula: https://en.wikipedia.org/wiki/Variance
    # Unbiased estimator: https://stats.stackexchange.com/questions/307537/unbiased-estimator-of-the-variance-of-the-sample-variance
    theta_component_variance_std_errors = []
    for i in range(len(mean_theta_estimate)):
        component_estimates = [estimate[i] for estimate in theta_estimates]
        second_central_moment = scipy.stats.moment(component_estimates, moment=4)
        fourth_central_moment = scipy.stats.moment(component_estimates, moment=4)
        N = len(theta_estimates)
        theta_component_variance_std_errors.append(
            np.sqrt(
                N
                * (
                    ((N) ** 2 - 3) * (second_central_moment) ** 2
                    + ((N - 1) ** 2) * fourth_central_moment
                )
                / ((N - 3) * (N - 2) * ((N - 1) ** 2))
            )
        )

    approximate_standard_errors = np.empty_like(empirical_var_normalized)
    for i, j in np.ndindex(approximate_standard_errors.shape):
        approximate_standard_errors[i, j] = max(
            theta_component_variance_std_errors[i],
            theta_component_variance_std_errors[j],
        )

    print(f"\nMean parameter estimate:\n{mean_theta_estimate}")
    print(f"\nEmpirical variance of parameter estimates:\n{empirical_var_normalized}")
    print(
        f"\nEmpirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):\n{approximate_standard_errors}"
    )
    print(
        f"\nMean adaptive sandwich variance estimate:\n{mean_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\nMean classical sandwich variance estimate:\n{mean_classical_sandwich_var_estimate}",
    )
    print(
        f"\nMedian adaptive sandwich variance estimate:\n{median_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\nMedian classical sandwich variance estimate:\n{median_classical_sandwich_var_estimate}",
    )
    print(
        f"\nAdaptive sandwich variance estimate std errors from empirical:\n{(mean_adaptive_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}",
    )
    print(
        f"\nClassical sandwich variance estimate std errors from empirical:\n{(mean_classical_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}",
    )
    print(
        f"\nAdaptive sandwich variance estimate elementwise standard deviations:\n{adaptive_sandwich_var_estimate_std_deviations}",
    )
    print(
        f"\nClassical sandwich variance estimate elementwise standard deviations:\n{classical_sandwich_var_estimate_std_deviations}",
    )
    print(
        f"\nAdaptive sandwich variance estimate elementwise mins:\n{adaptive_sandwich_var_estimate_mins}",
    )
    print(
        f"\nClassical sandwich variance estimate elementwise mins:\n{classical_sandwich_var_estimate_mins}",
    )
    print(
        f"\nAdaptive sandwich variance estimate elementwise maxes:\n{adaptive_sandwich_var_estimate_maxes}",
    )
    print(
        f"\nClassical sandwich variance estimate elementwise maxes:\n{classical_sandwich_var_estimate_maxes}\n",
    )

    if condition_numbers:
        print(
            f"\nMedian joint adaptive inverse bread condition number:\n{np.median(condition_numbers)}\n",
        )
        print(
            f"\nMinimum joint adaptive inverse bread condition number:\n{np.min(condition_numbers)}\n",
        )
        print(
            f"\nMaximum joint adaptive inverse bread condition number:\n{np.max(condition_numbers)}\n",
        )

    if condition_numbers_first_block:
        print(
            f"\nMedian joint adaptive inverse bread FIRST BLOCK condition number:\n{np.median(condition_numbers_first_block)}\n",
        )
        print(
            f"\nMinimum joint adaptive inverse bread FIRST BLOCK condition number:\n{np.min(condition_numbers_first_block)}\n",
        )
        print(
            f"\nMaximum joint adaptive inverse bread FIRST BLOCK condition number:\n{np.max(condition_numbers_first_block)}\n",
        )

    if theta_estimates[0].size == 1:
        index_to_check_ci_coverage = 0
    if index_to_check_ci_coverage is not None:
        # We take this to be the "true" value
        scalar_mean_theta = mean_theta_estimate[index_to_check_ci_coverage]
        diffs = np.abs(
            theta_estimates[:, index_to_check_ci_coverage] - scalar_mean_theta
        )

        adaptive_standard_errors = np.sqrt(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )
        classical_standard_errors = np.sqrt(
            classical_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )
        NOMINAL_COVERAGE = 0.95
        UPPER_PERCENTILE = 1 - (1 - NOMINAL_COVERAGE) / 2

        adaptive_z_covers = (
            diffs < scipy.stats.norm.ppf(UPPER_PERCENTILE) * adaptive_standard_errors
        )
        classical_z_covers = (
            diffs < scipy.stats.norm.ppf(UPPER_PERCENTILE) * classical_standard_errors
        )

        adaptive_t_covers = (
            diffs
            < scipy.stats.t.ppf(UPPER_PERCENTILE, num_users - 1)
            * adaptive_standard_errors
        )
        classical_t_covers = (
            diffs
            < scipy.stats.t.ppf(UPPER_PERCENTILE, num_users - 1)
            * classical_standard_errors
        )

        print(
            f"\nAdaptive sandwich {NOMINAL_COVERAGE * 100}% standard normal CI coverage:\n{np.mean(adaptive_z_covers)}\n",
        )
        print(
            f"\nClassical sandwich {NOMINAL_COVERAGE * 100}% standard normal CI coverage:\n{np.mean(classical_z_covers)}\n",
        )
        print(
            f"\nAdaptive sandwich {NOMINAL_COVERAGE * 100}% t({num_users - 1}) CI coverage:\n{np.mean(adaptive_t_covers)}\n",
        )
        print(
            f"\nClassical sandwich {NOMINAL_COVERAGE * 100}% t({num_users - 1}) CI coverage:\n{np.mean(classical_t_covers)}\n",
        )

        # Helpful for debugging large adaptive sandwich variance estimates.  Take the slurm job ID
        # from the filename, look at the logs, grab the seeds, and then can, e.g.,  use seed
        # overrides locally to examine behavior.
        if max_adaptive_estimate_at_index_filename is not None:
            print(
                f"\nMaximum adaptive sandwich variance estimate at index {index_to_check_ci_coverage} was {max_adaptive_estimate_at_index} in file {max_adaptive_estimate_at_index_filename}\n"
            )

        print("\nNow examining stability.\n")

        # Make sure previous output is flushed and not cleared
        sys.stdout.flush()
        plt.clear_terminal(False)

        # Plot the theta estimates to see variation
        plt.clear_figure()
        plt.title(f"Index {index_to_check_ci_coverage} of Theta Estimates")
        plt.xlabel("Simulation Index")
        plt.ylabel("Theta Estimate")
        plt.scatter(
            theta_estimates[:, index_to_check_ci_coverage],
            color="blue+",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(theta_estimates[:, index_to_check_ci_coverage]),
                max(1, len(theta_estimates[:, index_to_check_ci_coverage]) // 10),
            )
        )
        plt.show()

        # Plot the adaptive sandwich variance estimates to look for blowup
        plt.clear_figure()
        plt.title(
            f"Index {index_to_check_ci_coverage} of Adaptive Variance Estimates vs Empirical"
        )
        plt.xlabel("Simulation Index")
        plt.ylabel("Adaptive Variance Estimate")
        plt.scatter(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="green+",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(
                    adaptive_sandwich_var_estimates[
                        :, index_to_check_ci_coverage, index_to_check_ci_coverage
                    ]
                ),
                max(
                    1,
                    len(
                        adaptive_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    )
                    // 10,
                ),
            )
        )
        plt.horizontal_line(
            empirical_var_normalized[
                index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="red+",
        )
        plt.show()

        # Plot the classical sandwich variance estimates to look for blowup
        plt.clear_figure()
        plt.title(
            f"Index {index_to_check_ci_coverage} of Classical Variance Estimates vs Empirical"
        )
        plt.xlabel("Simulation Index")
        plt.ylabel("Classical Variance Estimate")
        plt.scatter(
            classical_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="green+",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(
                    classical_sandwich_var_estimates[
                        :, index_to_check_ci_coverage, index_to_check_ci_coverage
                    ]
                ),
                max(
                    1,
                    len(
                        classical_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    )
                    // 10,
                ),
            )
        )
        plt.horizontal_line(
            empirical_var_normalized[
                index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="red+",
        )
        plt.show()

        # Plot the adaptive sandwich standard error estimates to look for blowup
        plt.clear_figure()
        plt.title(
            f"Index {index_to_check_ci_coverage} of Adaptive SE Estimates vs Empirical"
        )
        plt.xlabel("Simulation Index")
        plt.ylabel("Adaptive SE Estimate")
        plt.scatter(
            np.sqrt(
                adaptive_sandwich_var_estimates[
                    :, index_to_check_ci_coverage, index_to_check_ci_coverage
                ]
            ),
            color="green+",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(
                    adaptive_sandwich_var_estimates[
                        :, index_to_check_ci_coverage, index_to_check_ci_coverage
                    ]
                ),
                max(
                    1,
                    len(
                        adaptive_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    )
                    // 10,
                ),
            )
        )
        plt.horizontal_line(
            np.sqrt(
                empirical_var_normalized[
                    index_to_check_ci_coverage, index_to_check_ci_coverage
                ]
            ),
            color="red+",
        )
        plt.show()

        # Plot the classical sandwich standard error estimates to look for blowup
        plt.clear_figure()
        plt.title(
            f"Index {index_to_check_ci_coverage} of Classical SE Estimates vs Empirical"
        )
        plt.xlabel("Simulation Index")
        plt.ylabel("Classical SE Estimate")
        plt.scatter(
            np.sqrt(
                classical_sandwich_var_estimates[
                    :, index_to_check_ci_coverage, index_to_check_ci_coverage
                ]
            ),
            color="green+",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(
                    classical_sandwich_var_estimates[
                        :, index_to_check_ci_coverage, index_to_check_ci_coverage
                    ]
                ),
                max(
                    1,
                    len(
                        classical_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    )
                    // 10,
                ),
            )
        )
        plt.horizontal_line(
            np.sqrt(
                empirical_var_normalized[
                    index_to_check_ci_coverage, index_to_check_ci_coverage
                ]
            ),
            color="red+",
        )
        plt.show()

        # Plot histogram of adaptive sandwich variance estimates
        plt.clear_figure()
        plt.title(
            "Histogram of Adaptive Sandwich Variance Estimates for Coefficient of Interest"
        )
        plt.xlabel("Adaptive Estimate")
        plt.ylabel("Frequency")
        plt.hist(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            bins=50,
            color="red+",
        )
        plt.grid(True)
        plt.show()

        # Plot histogram of classical sandwich variance estimates
        plt.clear_figure()
        plt.title(
            "Histogram of Classical Sandwich Variance Estimates for Coefficient of Interest"
        )
        plt.xlabel("Classical Estimate")
        plt.ylabel("Frequency")
        plt.hist(
            classical_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            bins=50,
            color="red+",
        )
        plt.grid(True)
        plt.show()

        plt.clear_figure()
        plt.title(
            "Overlaid Histograms of Adaptive(red) and Classical(blue) SEs for Coefficient of Interest"
        )
        # Compute the x range from the adaptive estimates
        adaptive_ses = np.sqrt(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )
        classical_ses = np.sqrt(
            classical_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )
        x_min, x_max = np.min(adaptive_ses), np.max(adaptive_ses)

        # Plot classical SEs histogram with adaptive x range
        plt.clear_figure()
        plt.title("Histogram of Classical SEs for Coefficient of Interest")
        plt.xlabel("Estimate")
        plt.ylabel("Frequency")
        plt.hist(classical_ses, bins=50, color="blue+")
        plt.xlim(x_min, x_max)
        plt.grid(True)
        plt.show()

        # Plot adaptive SEs histogram with same x range
        plt.clear_figure()
        plt.title("Histogram of Adaptive SEs for Coefficient of Interest")
        plt.xlabel("Estimate")
        plt.ylabel("Frequency")
        plt.hist(adaptive_ses, bins=50, color="red+")
        plt.xlim(x_min, x_max)
        plt.grid(True)
        plt.show()

        # Plot log classical SEs histogram with adaptive x range
        plt.clear_figure()
        plt.title("Histogram of Log Classical SEs for Coefficient of Interest")
        plt.xlabel("Log(Estimate)")
        plt.ylabel("Frequency")
        plt.hist(np.log(classical_ses), bins=50, marker="|", color="blue+")
        plt.xlim(np.log(x_min), np.log(x_max))
        plt.grid(True)
        plt.show()

        # Plot log adaptive SEs histogram with same x range
        plt.clear_figure()
        plt.title("Histogram of Log Adaptive SEs for Coefficient of Interest")
        plt.xlabel("Log(Estimate)")
        plt.ylabel("Frequency")
        plt.hist(np.log(adaptive_ses), bins=50, marker="|", color="red+")
        plt.xlim(np.log(x_min), np.log(x_max))
        plt.grid(True)
        plt.show()

        # Classical SEs (log scale)
        pyplt.figure(figsize=(8, 4))
        sns.histplot(np.log(classical_ses), color="blue", edgecolor="black")
        pyplt.title("Histogram of Log Classical SEs for Coefficient of Interest")
        pyplt.xlabel("Log(SE Estimate)")
        pyplt.ylabel("Frequency")
        pyplt.xlim(np.log(x_min), np.log(x_max))
        pyplt.grid(True, linestyle="--", alpha=0.6)
        # Add vertical line at log(sqrt(0.0378)), labeled "empirical log(SE)"
        pyplt.axvline(
            np.log(np.sqrt(0.0378)),
            color="green",
            linestyle="--",
            linewidth=2,
            label="empirical log(SE)",
        )
        pyplt.legend()
        pyplt.tight_layout()
        pyplt.savefig("classical_histogram_output.png", dpi=300, bbox_inches="tight")

        ### Print some fancier plots to files using seaborn/matplotlib

        # Adaptive SEs (log scale)
        pyplt.figure(figsize=(8, 4))
        sns.histplot(np.log(adaptive_ses), color="red", edgecolor="black")
        pyplt.title("Histogram of Log Adaptive SEs for Coefficient of Interest")
        pyplt.xlabel("Log(SE Estimate)")
        pyplt.ylabel("Frequency")
        pyplt.xlim(np.log(x_min), np.log(x_max))
        pyplt.grid(True, linestyle="--", alpha=0.6)
        # Add vertical line at log(sqrt(0.0378)), labeled "empirical log(SE)"
        pyplt.axvline(
            np.log(np.sqrt(0.0378)),
            color="green",
            linestyle="--",
            linewidth=2,
            label="empirical log(SE)",
        )
        pyplt.legend()
        pyplt.tight_layout()
        pyplt.savefig("adaptive_histogram_output.png", dpi=300, bbox_inches="tight")

        # Overlayed log-scale histograms of classical and adaptive SEs, mixing to purple where they overlap
        pyplt.figure(figsize=(8, 4))
        bins = np.linspace(np.log(x_min), np.log(x_max), 50)
        # Classical SEs histogram
        counts_classical, _, _ = pyplt.hist(
            np.log(classical_ses),
            bins=bins,
            color="blue",
            alpha=0.5,
            label="Log Classical SEs",
        )
        # Adaptive SEs histogram
        counts_adaptive, _, _ = pyplt.hist(
            np.log(adaptive_ses),
            bins=bins,
            color="red",
            alpha=0.5,
            label="Log Adaptive SEs",
        )
        # Where both histograms have nonzero counts, overlay a purple bar
        for i in range(len(bins) - 1):
            if counts_classical[i] > 0 and counts_adaptive[i] > 0:
                pyplt.bar(
                    (bins[i] + bins[i + 1]) / 2,
                    min(counts_classical[i], counts_adaptive[i]),
                    width=(bins[i + 1] - bins[i]),
                    color="purple",
                    alpha=0.7,
                    label="Overlap" if i == 0 else None,
                )
        pyplt.title(
            "Overlayed Histogram of Log SEs (Classical: Blue, Adaptive: Red, Overlap: Purple)"
        )
        pyplt.xlabel("Log(SE Estimate)")
        pyplt.ylabel("Frequency")
        pyplt.xlim(np.log(x_min), np.log(x_max))
        pyplt.grid(True, linestyle="--", alpha=0.6)
        pyplt.axvline(
            np.log(np.sqrt(0.0378)),
            color="green",
            linestyle="--",
            linewidth=2,
            label="empirical log(SE)",
        )
        pyplt.legend()
        pyplt.tight_layout()
        pyplt.savefig("overlayed_histogram_output.png", dpi=300, bbox_inches="tight")

        # Plot the classical sandwich variance estimates sorted by adaptive sandwich variance
        # estimates for the coefficient of interest
        num_experiments = max(1, len(adaptive_sandwich_var_estimates) * 5 // 100)
        # Get indices sorted by adaptive variance estimate (ascending)
        sorted_experiment_indices_by_adaptive_est = np.argsort(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )

        # Use sorted indices to split into two lists: those with adaptive variance > 5x empirical, and those <= 5x empirical
        empirical_var = empirical_var_normalized[
            index_to_check_ci_coverage, index_to_check_ci_coverage
        ]
        adaptive_var_at_index_sorted = adaptive_sandwich_var_estimates[
            sorted_experiment_indices_by_adaptive_est,
            index_to_check_ci_coverage,
            index_to_check_ci_coverage,
        ]
        # Find the point at which the adaptive variance estimate is more than
        # EMP_VAR_BLOWUP_MULTIPLIER times the empirical variance.
        EMP_VAR_BLOWUP_MULTIPLIER = 10
        estimate_blowup_split_idx = np.searchsorted(
            adaptive_var_at_index_sorted,
            EMP_VAR_BLOWUP_MULTIPLIER * empirical_var,
            side="right",
        )
        # Find where the empirical variance would fit into the sorted adaptive estimates
        empirical_variance_split_idx = np.searchsorted(
            adaptive_var_at_index_sorted, empirical_var, side="right"
        )

        print(
            f"\nNumber of simulations with adaptive variance estimate at index {index_to_check_ci_coverage} > {EMP_VAR_BLOWUP_MULTIPLIER}x empirical value: {len(adaptive_sandwich_var_estimates) - estimate_blowup_split_idx}\n"
        )
        print(
            f"Number of simulations with adaptive variance estimate at index {index_to_check_ci_coverage} > empirical value: {len(adaptive_sandwich_var_estimates) - empirical_variance_split_idx}\n"
        )

        classical_var_estimates_sorted_by_adaptive = classical_sandwich_var_estimates[
            sorted_experiment_indices_by_adaptive_est,
            index_to_check_ci_coverage,
            index_to_check_ci_coverage,
        ]

        plt.clear_figure()
        plt.title(
            f"Classical Estimates Sorted by Adaptive Variance Estimate at Index {index_to_check_ci_coverage} (vs. Classical Median) "
        )
        plt.xlabel("Experiment Index (sorted by Adaptive Variance)")
        plt.ylabel("Classical Variance Estimate")
        plt.scatter(
            classical_var_estimates_sorted_by_adaptive,
            color="orange",
        )
        plt.horizontal_line(
            median_classical_sandwich_var_estimate[
                index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="red+",
        )
        plt.xticks(range(1, num_experiments + 1, max(1, num_experiments // 10)))
        plt.show()

        if condition_numbers:

            condition_numbers = np.array(condition_numbers).astype(np.float64)

            # Plot histogram of joint bread inverse condition numbers
            plt.clear_figure()
            plt.title("Histogram of Joint Bread Inverse Condition Numbers")
            plt.xlabel("Condition Number")
            plt.ylabel("Frequency")
            plt.hist(condition_numbers, bins=20, color="purple")
            plt.grid(True)
            plt.show()

            condition_numbers_sorted_by_adaptive_est = [
                condition_numbers[i] for i in sorted_experiment_indices_by_adaptive_est
            ]

            min_condition_number_for_large_estimates = (
                np.min(condition_numbers[estimate_blowup_split_idx:])
                if len(condition_numbers[estimate_blowup_split_idx:]) > 0
                else None
            )
            print(
                f"\nMinimum joint bread inverse condition number for trials with adaptive variance estimate at index {index_to_check_ci_coverage} > {EMP_VAR_BLOWUP_MULTIPLIER}x empirical value: {min_condition_number_for_large_estimates}\n"
            )

            plt.clear_figure()
            plt.title(
                f"Joint Bread Inverse Condition Numbers Sorted by Adaptive Variance Estimate at Index {index_to_check_ci_coverage}. Emp var insert idx in green, {EMP_VAR_BLOWUP_MULTIPLIER}x emp var insert idx in red."
            )
            plt.xlabel("Experiment Index (sorted by Adaptive Variance)")
            plt.ylabel("Condition Number")
            # Plot all sorted condition numbers and a threshold line after which
            # adaptive variance is > 5x empirical value
            plt.scatter(
                condition_numbers_sorted_by_adaptive_est,
                color="blue+",
            )
            plt.vertical_line(
                empirical_variance_split_idx,
                color="green+",
            )
            plt.vertical_line(
                estimate_blowup_split_idx,
                color="red+",
            )
            plt.xticks(
                range(
                    0,
                    len(condition_numbers),
                    max(1, len(condition_numbers) // 10),
                )
            )
            plt.grid(True)
            plt.show()

            # Plot the adaptive sandwich variance estimates to look for blowup, compared with condition numbers
            plt.clear_figure()
            plt.title(
                f"Index {index_to_check_ci_coverage} of Adaptive Variance Estimates (green) vs Empirical w/ Joint Adaptive Bread Inv Condition Numbers (blue)"
            )
            plt.xlabel("Simulation Index")
            plt.ylabel("Adaptive Variance Estimate")
            plt.scatter(
                adaptive_sandwich_var_estimates[
                    :, index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="green+",
            )
            plt.scatter(
                condition_numbers,
                color="blue+",
                yside="right",
            )
            plt.grid(True)
            plt.xticks(
                range(
                    0,
                    len(
                        adaptive_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    ),
                    max(
                        1,
                        len(
                            adaptive_sandwich_var_estimates[
                                :,
                                index_to_check_ci_coverage,
                                index_to_check_ci_coverage,
                            ]
                        )
                        // 10,
                    ),
                )
            )
            plt.horizontal_line(
                empirical_var_normalized[
                    index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="red+",
            )
            plt.show()

        if condition_numbers_first_block:

            condition_numbers_first_block = np.array(
                condition_numbers_first_block
            ).astype(np.float64)

            # Plot histogram of joint bread inverse first block condition numbers
            plt.clear_figure()
            plt.title("Histogram of Joint Bread Inverse First Block Condition Numbers")
            plt.xlabel("Condition Number")
            plt.ylabel("Frequency")
            plt.hist(condition_numbers_first_block, bins=20, color="purple")
            plt.grid(True)
            plt.show()

            condition_numbers_first_block_sorted_by_adaptive_est = [
                condition_numbers_first_block[i]
                for i in sorted_experiment_indices_by_adaptive_est
            ]

            min_condition_number_first_block_for_large_estimates = (
                np.min(condition_numbers_first_block[estimate_blowup_split_idx:])
                if len(condition_numbers_first_block[estimate_blowup_split_idx:]) > 0
                else None
            )
            print(
                f"\nMinimum joint bread inverse FIRST BLOCK condition number for trials with adaptive variance estimate at index {index_to_check_ci_coverage} > {EMP_VAR_BLOWUP_MULTIPLIER}x empirical value: {min_condition_number_first_block_for_large_estimates}\n"
            )

            plt.clear_figure()
            plt.title(
                f"Joint Bread Inverse First Block Condition Numbers Sorted by Adaptive Variance Estimate at Index {index_to_check_ci_coverage}. Emp var insert idx in green, {EMP_VAR_BLOWUP_MULTIPLIER}x emp var insert idx in red."
            )
            plt.xlabel("Experiment Index (sorted by Adaptive Variance)")
            plt.ylabel("Condition Number")
            # Plot all sorted condition numbers and a threshold line after which
            # adaptive variance is > 5x empirical value
            plt.scatter(
                condition_numbers_first_block_sorted_by_adaptive_est,
                color="blue+",
            )
            plt.vertical_line(
                empirical_variance_split_idx,
                color="green+",
            )
            plt.vertical_line(
                estimate_blowup_split_idx,
                color="red+",
            )
            plt.xticks(
                range(
                    0,
                    len(condition_numbers_first_block),
                    max(1, len(condition_numbers_first_block) // 10),
                )
            )
            plt.grid(True)
            plt.show()

            # Plot the adaptive sandwich variance estimates to look for blowup, compared with first block condition numbers
            plt.clear_figure()
            plt.title(
                f"Index {index_to_check_ci_coverage} of Adaptive Variance Estimates (green) vs Empirical w/ First Block Condition Numbers (blue)"
            )
            plt.xlabel("Simulation Index")
            plt.ylabel("Adaptive Variance Estimate")
            plt.scatter(
                adaptive_sandwich_var_estimates[
                    :, index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="green+",
            )
            plt.scatter(
                condition_numbers_first_block,
                color="blue+",
                yside="right",
            )
            plt.grid(True)
            plt.xticks(
                range(
                    0,
                    len(
                        adaptive_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    ),
                    max(
                        1,
                        len(
                            adaptive_sandwich_var_estimates[
                                :,
                                index_to_check_ci_coverage,
                                index_to_check_ci_coverage,
                            ]
                        )
                        // 10,
                    ),
                )
            )
            plt.horizontal_line(
                empirical_var_normalized[
                    index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="red+",
            )
            plt.show()

        if identity_diff_abs_maxes:

            identity_diff_abs_maxes_sorted_by_adaptive_est = [
                identity_diff_abs_maxes[i]
                for i in sorted_experiment_indices_by_adaptive_est
            ]

            plt.clear_figure()
            plt.title(
                f"Abs Max of Identity Diff Sorted by Adaptive Var Estimate at Index {index_to_check_ci_coverage}. Emp var insert idx in green, {EMP_VAR_BLOWUP_MULTIPLIER}x emp var insert idx in red."
            )
            plt.xlabel("Experiment Index (sorted by Adaptive Variance)")
            plt.ylabel("Abs Max of Identity Diff")
            plt.scatter(
                identity_diff_abs_maxes_sorted_by_adaptive_est,
                color="blue+",
            )
            plt.vertical_line(
                empirical_variance_split_idx,
                color="green+",
            )
            plt.vertical_line(
                estimate_blowup_split_idx,
                color="red+",
            )
            plt.xticks(
                range(
                    0,
                    len(identity_diff_abs_maxes),
                    max(1, len(identity_diff_abs_maxes) // 10),
                )
            )
            plt.grid(True)
            plt.show()

            # Plot the adaptive sandwich variance estimates to look for blowup, compared with identity diff abs maxes
            plt.clear_figure()
            plt.title(
                f"Index {index_to_check_ci_coverage} of Adaptive Variance Estimates (green) vs Empirical w/ Identity Diff Abs Maxes (blue)"
            )
            plt.xlabel("Simulation Index")
            plt.ylabel("Adaptive Variance Estimate")
            plt.scatter(
                adaptive_sandwich_var_estimates[
                    :, index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="green+",
            )
            plt.scatter(
                identity_diff_abs_maxes,
                color="blue+",
                yside="right",
            )
            plt.grid(True)
            plt.xticks(
                range(
                    0,
                    len(
                        adaptive_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    ),
                    max(
                        1,
                        len(
                            adaptive_sandwich_var_estimates[
                                :,
                                index_to_check_ci_coverage,
                                index_to_check_ci_coverage,
                            ]
                        )
                        // 10,
                    ),
                )
            )
            plt.horizontal_line(
                empirical_var_normalized[
                    index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="red+",
            )
            plt.show()

        if identity_diff_frobenius_norms:

            identity_diff_frobenius_norms_sorted_by_adaptive_est = [
                identity_diff_frobenius_norms[i]
                for i in sorted_experiment_indices_by_adaptive_est
            ]

            plt.clear_figure()
            plt.title(
                f"Frobenius Norm of Identity Diff Sorted by Adaptive Var Estimate at Index {index_to_check_ci_coverage}. Emp var insert idx in green, {EMP_VAR_BLOWUP_MULTIPLIER}x emp var insert idx in red."
            )
            plt.xlabel("Experiment Index (sorted by Adaptive Variance)")
            plt.ylabel("Frobenius Norm of Identity Diff")
            plt.scatter(
                identity_diff_frobenius_norms_sorted_by_adaptive_est,
                color="blue+",
            )
            plt.vertical_line(
                empirical_variance_split_idx,
                color="green+",
            )
            plt.vertical_line(
                estimate_blowup_split_idx,
                color="red+",
            )
            plt.xticks(
                range(
                    0,
                    len(identity_diff_frobenius_norms),
                    max(1, len(identity_diff_frobenius_norms) // 10),
                )
            )
            plt.grid(True)
            plt.show()

            # Plot the adaptive sandwich variance estimates to look for blowup, compared with identity diff frobenius norms
            plt.clear_figure()
            plt.title(
                f"Index {index_to_check_ci_coverage} of Adaptive Variance Estimates (green) vs Empirical w/ Identity Diff Frobenius Norms (blue)"
            )
            plt.xlabel("Simulation Index")
            plt.ylabel("Adaptive Variance Estimate")
            plt.scatter(
                adaptive_sandwich_var_estimates[
                    :, index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="green+",
            )
            plt.scatter(
                identity_diff_frobenius_norms,
                color="blue+",
                yside="right",
            )
            plt.grid(True)
            plt.xticks(
                range(
                    0,
                    len(
                        adaptive_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    ),
                    max(
                        1,
                        len(
                            adaptive_sandwich_var_estimates[
                                :,
                                index_to_check_ci_coverage,
                                index_to_check_ci_coverage,
                            ]
                        )
                        // 10,
                    ),
                )
            )
            plt.horizontal_line(
                empirical_var_normalized[
                    index_to_check_ci_coverage, index_to_check_ci_coverage
                ],
                color="red+",
            )
            plt.show()

        # Examine conditioning of first block of joint bread inverse if the data is available
        if len(min_eigvals_first_block) > 0 and len(max_eigvals_first_block) > 0:
            min_eigenvalues_first_block = np.array(min_eigvals_first_block)
            max_eigenvalues_first_block = np.array(max_eigvals_first_block)

            # Plot histogram of minimum eigenvalues for first diagonal block of joint bread inverse
            plt.clear_figure()
            plt.title(
                "Histogram of Minimum Eigenvalues of Joint Bread Inverse Matrix First Diag Block"
            )
            plt.xlabel("Minimum Eigenvalue")
            plt.ylabel("Frequency")
            plt.hist(min_eigenvalues_first_block, bins=20, color="green+")
            plt.grid(True)
            plt.show()

            # Plot histogram of maximum eigenvalues for first diagonal block of joint bread inverse
            plt.clear_figure()
            plt.title(
                "Histogram of Maximum Eigenvalues of Joint Bread Inverse Matrix First Diag Block"
            )
            plt.xlabel("Maximum Eigenvalue")
            plt.ylabel("Frequency")
            plt.hist(max_eigenvalues_first_block, bins=20, color="orange")
            plt.grid(True)
            plt.show()

            sorted_min_eigenvalues_first_block = np.array(
                [
                    min_eigenvalues_first_block[i]
                    for i in sorted_experiment_indices_by_adaptive_est
                ]
            )
            # Plot minimum eigenvalues for first diagonal block of joint bread inverse
            plt.clear_figure()
            plt.title(
                f"Minimum Eigenvalues of Joint Bread Inverse Matrix First Diag Block Sorted by Adaptive Variance Estimate at Index {index_to_check_ci_coverage}"
            )
            plt.xlabel("Simulation Index (sorted by Adaptive Variance)")
            plt.ylabel("Min Eigenvalue")
            plt.scatter(sorted_min_eigenvalues_first_block, color="orange")
            plt.grid(True)
            plt.xticks(
                range(
                    0,
                    len(min_eigenvalues_first_block),
                    max(1, len(min_eigenvalues_first_block) // 10),
                )
            )
            plt.show()

            sorted_max_eigenvalues_first_block = np.array(
                [
                    max_eigenvalues_first_block[i]
                    for i in sorted_experiment_indices_by_adaptive_est
                ]
            )
            # Plot maximum eigenvalues for first diagonal block of joint bread inverse
            plt.clear_figure()
            plt.title(
                f"Maximum Eigenvalues of Joint Bread Inverse Matrix First Diag Block Sorted by Adaptive Variance Estimate at Index {index_to_check_ci_coverage}"
            )
            plt.xlabel("Simulation Index (sorted by Adaptive Variance)")
            plt.ylabel("Max Eigenvalue")
            plt.scatter(sorted_max_eigenvalues_first_block, color="orange")
            plt.grid(True)
            plt.xticks(
                range(
                    0,
                    len(max_eigenvalues_first_block),
                    max(1, len(max_eigenvalues_first_block) // 10),
                )
            )
            plt.show()

        # Examine normality of first beta if available
        if len(first_beta_coords) > 0:
            first_beta_coords_arr = np.array(first_beta_coords)
            for coordinate in range(min(5, first_beta_coords_arr.shape[1])):
                plt.clear_figure()
                plt.title(f"Histogram of First Update Beta Coordinate {coordinate}")
                plt.xlabel("First Update Beta Coordinate")
                plt.ylabel("Frequency")
                plt.hist(
                    first_beta_coords_arr[:, coordinate],
                    bins=20,
                    color="blue+",
                )
                plt.grid(True)
                plt.show()

        # Plot all action_1_fractions
        plt.clear_figure()
        plt.title("Action 1 Fractions for All Simulations")
        plt.xlabel("Simulation Index")
        plt.ylabel("Action 1 Fraction")
        plt.scatter(action_1_fractions, color="red+")
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(action_1_fractions),
                max(1, len(action_1_fractions) // 10),
            )
        )
        plt.show()

        # Plot action_1_fractions for the top 5% of adaptive variance estimates
        sorted_action_1_fractions = [
            action_1_fractions[i] for i in sorted_experiment_indices_by_adaptive_est
        ]
        plt.clear_figure()
        plt.title(
            f"Action 1 Fractions Sorted by Adaptive Variance Estimate at Index {index_to_check_ci_coverage}"
        )
        plt.xlabel("Experiment Index (sorted by Adaptive Variance)")
        plt.ylabel("Action 1 Fraction")
        plt.scatter(sorted_action_1_fractions, color="red+")
        plt.xticks(
            range(
                0,
                len(action_1_fractions),
                max(1, len(action_1_fractions) // 10),
            )
        )
        plt.grid(True)
        plt.show()

        # Plot action probability variances sorted by size of the adaptive variance
        sorted_action_prob_variances = [
            action_prob_variances[i] for i in sorted_experiment_indices_by_adaptive_est
        ]
        plt.clear_figure()
        plt.title(
            f"Action Probability Variances Sorted by Adaptive Variance Estimate at Index {index_to_check_ci_coverage}"
        )
        plt.xlabel("Experiment Index (sorted by Adaptive Variance)")
        plt.ylabel("Action Probability Variance")
        plt.scatter(sorted_action_prob_variances, color="blue+")
        plt.xticks(
            range(
                0,
                len(action_prob_variances),
                max(1, len(action_prob_variances) // 10),
            )
        )
        plt.grid(True)
        plt.show()
