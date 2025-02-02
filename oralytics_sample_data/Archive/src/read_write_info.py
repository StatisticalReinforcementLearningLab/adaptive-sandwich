import pathlib

READ_PATH_PREFIX = str(pathlib.Path(__file__).parent.parent) + "/"
WRITE_PATH_PREFIX = READ_PATH_PREFIX + "exps/write/"

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
