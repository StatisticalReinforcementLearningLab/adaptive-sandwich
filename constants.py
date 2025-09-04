class RLStudyArgs:
    """Simple container for centralizing the strings used by the RL algorithm"""

    # Dataset Type
    HEARTSTEPS = "heartsteps"
    SYNTHETIC = "synthetic"
    ORALYTICS = "oralytics"

    # Heartsteps Mode
    EVALSIM = "evalSim"
    REALISTIC = "realistic"
    MEDIUM = "medium"
    EASY = "easy"

    # Synthetic mode
    DELAYED_1_ACTION_DOSAGE = "delayed_1_action_dosage"
    DELAYED_2_ACTION_DOSAGE = "delayed_2_action_dosage"
    DELAYED_5_ACTION_DOSAGE = "delayed_5_action_dosage"
    DELAYED_1_DOSAGE_PAPER = "delayed_1_dosage_paper"
    DELAYED_2_DOSAGE_PAPER = "delayed_2_dosage_paper"
    DELAYED_5_DOSAGE_PAPER = "delayed_5_dosage_paper"

    # RL Algorithm
    FIXED_RANDOMIZATION = "fixed_randomization"
    SIGMOID_LS_HARD_CLIP = "sigmoid_LS_hard_clip"
    SIGMOID_LS_SMOOTH_CLIP = "sigmoid_LS_smooth_clip"
    SMOOTH_POSTERIOR_SAMPLING = "smooth_posterior_sampling"

    # Noise error correlation
    TIME_CORR = "time_corr"
    INDEPENDENT = "independent"

    # Algorithm state features
    INTERCEPT = "intercept"
    PAST_REWARD = "past_reward"
    TIME_OF_DAY = "time_of_day"
    PRIOR_DAY_BRUSH = "prior_day_brush"
    WEEKEND = "weekend"
    DAY_IN_STUDY_NORM = "day_in_study_norm"

    # Prior for posterior sampling
    NAIVE = "naive"
    ORALYTICS = "oralytics"

    # Additional args with defaults set per algorithm
    T = "T"
    RECRUIT_N = "recruit_n"
    RECRUIT_T = "recruit_t"


class SmallSampleCorrections:
    NONE = "none"
    HC1theta = "HC1theta"
    HC2theta = "HC2theta"
    HC3theta = "HC3theta"


class InverseStabilizationMethods:
    NONE = "none"
    TRIM_SMALL_SINGULAR_VALUES = "trim_small_singular_values"
    ZERO_OUT_SMALL_OFF_DIAGONALS = "zero_out_small_off_diagonals"
    ADD_RIDGE_FIXED_CONDITION_NUMBER = "add_ridge_fixed_condition_number"
    ADD_RIDGE_MEDIAN_SINGULAR_VALUE_FRACTION = (
        "add_ridge_median_singular_value_fraction"
    )
    INVERSE_BREAD_STRUCTURE_AWARE_INVERSION = "inverse_bread_structure_aware_inversion"
    ALL_METHODS_COMPETITION = "all_methods_competition"


class FunctionTypes:
    LOSS = "loss"
    ESTIMATING = "estimating"


class SandwichFormationMethods:
    BREAD_INVERSE_T_QR = "bread_inverse_T_qr"
    MEAT_SVD_SOLVE = "meat_svd_solve"
    NAIVE = "naive"
