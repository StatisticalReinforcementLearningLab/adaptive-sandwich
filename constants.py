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
    SIGMOID_LS = "sigmoid_LS"
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
    none = "none"
    HC1 = "HC1"
    HC2 = "HC2"
    HC3 = "HC3"


class FunctionTypes:
    LOSS = "loss"
    ESTIMATING = "estimating"
