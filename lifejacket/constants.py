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
