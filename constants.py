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
    DELAYED_EFFECTS = "delayed_effects"

    # RL Algorithm
    FIXED_RANDOMIZATION = "fixed_randomization"
    SIGMOID_LS = "sigmoid_LS"
    POSTERIOR_SAMPLING = "posterior_sampling"

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
    ALLOCATION_SIGMA = "allocation_sigma"
    NOISE_VAR = "noise_var"
