### TODO: fill in with your specific read and write path
# READ_PATH_PREFIX = "/Users/sghosh/Research/oralytics_algorithm_design/"
# WRITE_PATH_PREFIX = "/Users/sghosh/Research/oralytics_algorithm_design/exps/read/"

READ_PATH_PREFIX = (
    "/Users/nowellclosser/code/adaptive-sandwich/oralytics_sample_data/Archive/"
)
WRITE_PATH_PREFIX = "/Users/nowellclosser/code/adaptive-sandwich/oralytics_sample_data/Archive/exps/write/"

exp_kwargs = {
    "sim_env_version": "v3",
    "base_env_type": "NON_STAT",
    "effect_size_scale": "None",
    "delayed_effect_scale": "LOW_R",
    "alg_type": "BLR_AC_V3",
    "noise_var": "None",
    "clipping_vals": [0.2, 0.8],
    "b_logistic": 0.515,
    "update_cadence": 14,
    "cluster_size": "full_pooling",
    "cost_params": [80, 40],
}
