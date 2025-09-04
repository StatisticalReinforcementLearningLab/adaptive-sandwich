import argparse
import glob
import logging
import re
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import mode
from sklearn.metrics import auc, f1_score, r2_score, roc_curve
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # headless backend

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

EMPIRICAL_TRACE = 0.15124110  # Trace of Empirical Covariance Matrix

parser = argparse.ArgumentParser(description="Train XGBoost models on pickled data.")
parser.add_argument(
    "--input_glob", type=str, required=True, help="Glob pattern for input pickle files."
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save model outputs."
)
parser.add_argument(
    "--empirical_trace_blowup_factor",
    type=float,
    required=True,
    help="The factor by which the empirical trace is multiplied to define the blowup threshold.",
)


def regression_label_type(value):
    if value in ["log1p_trace", "raw_trace"]:
        return value
    if value.startswith("index_"):
        try:
            int(value.split("_")[1])
            return value
        except (IndexError, ValueError) as e:
            raise argparse.ArgumentTypeError from e(
                f"Invalid index_x format: {value}. Must be index_<int>."
            )
    raise argparse.ArgumentTypeError(
        f"Invalid regression_label_type: {value}. Must be 'log1p_trace', 'raw_trace', or 'index_<int>'."
    )


parser.add_argument(
    "--regression_label_type",
    type=regression_label_type,
    default="log1p_trace",
    help="Label transformation for regression. Choices: 'log1p_trace', 'raw_trace', or 'index_<int>'.",
)


def classification_label_type(value):
    if value in ["trace_blowup"] or re.match(r"^index_\d+_blowup$", value):
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid classification_label_type: {value}. Must be 'trace_blowup', or 'index_<int>_blowup'."
    )


parser.add_argument(
    "--classification_label_type",
    type=classification_label_type,
    default="trace_blowup",
    help="Label transformation for classification. Choices: 'trace_blowup', 'index_<int>_blowup'.",
)

args = parser.parse_args()

INPUT_GLOB = args.input_glob
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------------------------------------
# 1. Collect file paths
# ---------------------------------------------------------------------

file_list = sorted(glob.glob(INPUT_GLOB, recursive=True))
if not file_list:
    raise RuntimeError("No pickle files found. Check INPUT_GLOB.")

logger.info("Found %d pickled dicts", len(file_list))

# ---------------------------------------------------------------------
# 2. Load pickles -> list of dicts
# ---------------------------------------------------------------------
dicts = []
labels = []
for path in file_list:
    with open(path, "rb") as f:
        d = pickle.load(f)
        label = d.pop("label")
    dicts.append(d)
    labels.append(label)

logger.info("Loaded all dictionaries into memory")

# ---------------------------------------------------------------------
# 3. Build DataFrame  (label column must be numeric)
# ---------------------------------------------------------------------
df = pd.DataFrame(dicts)
labels = np.array(labels, dtype="float32")

# Ensure numeric dtypes
df = df.astype("float32")


# Determine appropriate label transformations for regression
if args.regression_label_type == "log1p_trace":
    y = np.log1p(np.trace(labels, axis1=1, axis2=2).astype("float32"))
elif args.regression_label_type == "raw_trace":
    y = np.trace(labels, axis1=1, axis2=2).astype("float32")
elif args.regression_label_type.startswith("index_"):
    index = int(args.regression_label_type.split("_")[1])
    if index < 0 or index >= labels.shape[1]:
        raise ValueError(
            f"Index {index} out of bounds for label array with shape {labels.shape}"
        )
    y = labels[:, index, index].astype("float32")
else:
    raise ValueError("Invalid regression_label_type")

# Determine appropriate label transformations for classification
if args.classification_label_type == "trace_blowup":
    traces = np.trace(labels, axis1=1, axis2=2)
    y_binary = (traces > args.empirical_trace_blowup_factor * EMPIRICAL_TRACE).astype(
        "int32"
    )
elif args.classification_label_type.startswith("index_"):
    index = int(args.classification_label_type.split("_")[1])
    if index < 0 or index >= labels.shape[1]:
        raise ValueError(
            f"Index {index} out of bounds for label array with shape {labels.shape}"
        )
    y_binary = (
        labels[:, index, index] > args.empirical_trace_blowup_factor * EMPIRICAL_TRACE
    ).astype("int32")
else:
    raise ValueError("Invalid classification_label_type")

X = df

# Also create a version of X with the best predictors removed.
X_no_overall_cond_and_min_singular_value = df.copy()
X_no_overall_cond_and_min_singular_value.pop("joint_bread_inverse_condition_number")
X_no_overall_cond_and_min_singular_value.pop("joint_bread_inverse_min_singular_value")

X_no_premature_adaptive_sandwich_features = df.copy()
for col in list(X_no_premature_adaptive_sandwich_features.columns):
    if col.startswith("premature_"):
        X_no_premature_adaptive_sandwich_features.pop(col)

logger.info("Shape after load  ->  X: %s,  y: %s", X.shape, y.shape)


# Include premature adaptive sandwich estimates only up to certain update numbers.
# Note that this filters out the condition numbers and classical estimates,
# which don't seem to be that important.
X_premature_adaptive_sandwich_features_up_to_max_update = {}
for max_update_num in (2, 3, 4):
    X_premature_adaptive_sandwich_features_up_to_max_update[max_update_num] = df.copy()
    for col in list(
        X_premature_adaptive_sandwich_features_up_to_max_update[max_update_num].columns
    ):
        if col.startswith("premature_"):
            match = re.match(r"^premature_adaptive_sandwich_update_(\d+)_", col)
            if match:
                # Note the + 1 here because the updates are 0-indexed in the feature names
                update_num = int(match.group(1)) + 1
                if update_num > max_update_num:
                    X_premature_adaptive_sandwich_features_up_to_max_update[
                        max_update_num
                    ].pop(col)
            else:
                X_premature_adaptive_sandwich_features_up_to_max_update[
                    max_update_num
                ].pop(col)

# ---------------------------------------------------------------------
# 4. Train / validation split
# ---------------------------------------------------------------------
# First split off the validation set (30 % of all data)
# Stratify by the true binary blowup label since the dataset is imbalanced.
X_train, X_val, y_train, y_val = train_test_split(
    X.values, y, test_size=0.3, random_state=42, stratify=y_binary
)


feature_names = X.columns.tolist()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)


# Set up a second training set with the best predictors removed.
# This is to see if we can get decent performance without the premature adaptive sandwich features.
# Note that the labels are the same as before, we can reuse them. Just generate new features.
X_trunc_train, X_trunc_val, _, _ = train_test_split(
    X_no_premature_adaptive_sandwich_features.values,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y_binary,
)

feature_names_trunc = X_no_premature_adaptive_sandwich_features.columns.tolist()
dtrain_trunc = xgb.DMatrix(
    X_trunc_train, label=y_train, feature_names=feature_names_trunc
)
dval_trunc = xgb.DMatrix(X_trunc_val, label=y_val, feature_names=feature_names_trunc)

# Set up a binary classification version of the training and validation sets.
# Note only the labels are new, can reuse the features from before.
_, _, y_train_binary, y_val_binary = train_test_split(
    X.values, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)
dtrain_binary = xgb.DMatrix(X_train, label=y_train_binary, feature_names=feature_names)
dval_binary = xgb.DMatrix(X_val, label=y_val_binary, feature_names=feature_names)

# Set up a binary classification version of the training and validation sets with best predictors removed.
dtrain_binary_trunc = xgb.DMatrix(
    X_trunc_train, label=y_train_binary, feature_names=feature_names_trunc
)
dval_binary_trunc = xgb.DMatrix(
    X_trunc_val, label=y_val_binary, feature_names=feature_names_trunc
)


# Set up binary classification training and validation sets only allowed to use
# premature adaptive sandwich features up to certain update numbers.
X_partial_premature_features_training_sets = {}
X_partial_premature_features_validation_sets = {}
dtrain_dval_tuples_for_partial_premature_features = {}
for (
    max_update_num,
    X_subset,
) in X_premature_adaptive_sandwich_features_up_to_max_update.items():
    (
        X_partial_premature_features_training_sets[max_update_num],
        X_partial_premature_features_validation_sets[max_update_num],
        _,
        _,
    ) = train_test_split(
        X_subset.values, y, test_size=0.3, random_state=42, stratify=y_binary
    )
    feature_names_partial = X_subset.columns.tolist()
    dtrain_dval_tuples_for_partial_premature_features[max_update_num] = (
        xgb.DMatrix(
            X_partial_premature_features_training_sets[max_update_num],
            label=y_train_binary,
            feature_names=feature_names_partial,
        ),
        xgb.DMatrix(
            X_partial_premature_features_validation_sets[max_update_num],
            label=y_val_binary,
            feature_names=feature_names_partial,
        ),
    )

# ---------------------------------------------------------------------
# 5. XGBoost parameters and training
# ---------------------------------------------------------------------

watchlist = [(dtrain, "train"), (dval, "val")]
watchlist_trunc = [(dtrain_trunc, "train"), (dval_trunc, "val")]
watchlist_binary = [(dtrain_binary, "train"), (dval_binary, "val")]
watchlist_binary_trunc = [(dtrain_binary_trunc, "train"), (dval_binary_trunc, "val")]

# default learning rate and regularization
params_1 = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
}
logger.info("Training model 1, with default parameters...")
model_1 = xgb.train(
    params_1,
    dtrain,
    num_boost_round=2000,
    evals=watchlist,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

# logger.info(
#     "Training model 2, with default parameters like model 1 but no early stopping..."
# )
# model_2 = xgb.train(
#     params_1,
#     dtrain,
#     num_boost_round=2000,
#     evals=watchlist,
#     verbose_eval=100,
# )

params_2 = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,  # small steps to avoid overfitting, not so small to drag out training
    "max_depth": 4,  # shallow trees to limit overfitting
    "subsample": 0.8,  # makes each tree see only a subset of data
    "colsample_bytree": 0.35,  # make only a subset of features available to each tree
    "reg_lambda": 2.0,  # moderate ridge to penalize leaf weights
    "reg_alpha": 0.1,  # light lasso to encourage some sparsity
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
}

logger.info("Training model 3, with more regularization and tuned learning rate...")
model_3 = xgb.train(
    params_2,
    dtrain,
    num_boost_round=2000,
    evals=watchlist,
    # if validation metric doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

# logger.info(
#     "Training model 4, with more regularization and tuned learning rate like model 2 but no early stopping"
# )
# model_4 = xgb.train(
#     params_2,
#     dtrain,
#     num_boost_round=2000,
#     evals=watchlist,
#     verbose_eval=100,
# )


logger.info("Training model 5, with truncated feature set...")
model_5 = xgb.train(
    params_1,
    dtrain_trunc,
    num_boost_round=2000,
    evals=watchlist_trunc,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

# logger.info(
#     "Training model 6, with truncated feature set like model 5 but with no early stopping..."
# )
# model_6 = xgb.train(
#     params_1,
#     dtrain_trunc,
#     num_boost_round=2000,
#     evals=watchlist_trunc,
#     verbose_eval=100,
# )

logger.info("Training model 7, with truncated feature set and more regularization...")
model_7 = xgb.train(
    params_2,
    dtrain_trunc,
    num_boost_round=2000,
    evals=watchlist_trunc,
    # if validation metric doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)


# logger.info(
#     "Training model 8, with truncated feature set and more regularization like model 7 but with no early stopping..."
# )
# model_8 = xgb.train(
#     params_2,
#     dtrain_trunc,
#     num_boost_round=2000,
#     evals=watchlist_trunc,
#     verbose_eval=100,
# )

regression_models_and_data = [
    (model_1, dtrain, dval, "continuous outcome, default parameters, early stopping"),
    # (
    #     model_2,
    #     dtrain,
    #     dval,
    #     "continuous outcome, default parameters, no early stopping",
    # ),
    (
        model_3,
        dtrain,
        dval,
        "continuous outcome, more regularization and tuned learning rate, early stopping",
    ),
    # (
    #     model_4,
    #     dtrain,
    #     dval,
    #     "continuous outcome, more regularization and tuned learning rate, no early stopping",
    # ),
    (
        model_5,
        dtrain_trunc,
        dval_trunc,
        "continuous outcome, truncated feature set, default parameters, early stopping",
    ),
    # (
    #     model_6,
    #     dtrain_trunc,
    #     dval_trunc,
    #     "continuous outcome, truncated feature set, no early stopping",
    # ),
    (
        model_7,
        dtrain_trunc,
        dval_trunc,
        "continuous outcome, truncated feature set, more regularization, early stopping",
    ),
    # (
    #     model_8,
    #     dtrain_trunc,
    #     dval_trunc,
    #     "continuous outcome, truncated feature set, more regularization, no early stopping",
    # ),
]

params_3 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
}
params_4 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,  # small steps to avoid overfitting
    "max_depth": 4,  # shallow trees to limit overfitting
    "subsample": 0.8,  # makes each tree see only a subset of data
    "colsample_bytree": 0.35,  # make only a subset of features available to each tree
    "reg_lambda": 2.0,  # moderate ridge to penalize leaf weights
    "reg_alpha": 0.1,  # light lasso to encourage some sparsity
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
}
params_5 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,  # small steps to avoid overfitting
    "max_depth": 4,  # shallow trees to limit overfitting
    "subsample": 0.8,  # makes each tree see only a subset of data
    "colsample_bytree": 0.35,  # make only a subset of features available to each tree
    "reg_lambda": 2.0,  # moderate ridge to penalize leaf weights
    "reg_alpha": 0.1,  # light lasso to encourage some sparsity
    "tree_method": "hist",
    "device": "cuda",
    "seed": 42,
    # adjust for class imbalance
    "scale_pos_weight": sum(y_train_binary == 0) / sum(y_train_binary == 1),
    # at the same time, disallow giant updates to leaf weights
    # which can happen especially when the scale_pos_weight is large
    "max_delta_step": 1,
}

logger.info("Training model 9, binary classification with default parameters...")
model_9 = xgb.train(
    params_3,
    dtrain_binary,
    num_boost_round=2000,
    evals=watchlist_binary,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

logger.info(
    "Training model 10, binary classification with more regularization and tuned learning rate..."
)
model_10 = xgb.train(
    params_4,
    dtrain_binary,
    num_boost_round=2000,
    evals=watchlist_binary,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

logger.info(
    "Training model 11, binary classification with more regularization and tuned learning rate, scale_pos_weight..."
)
model_11 = xgb.train(
    params_5,
    dtrain_binary,
    num_boost_round=2000,
    evals=watchlist_binary,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds
    # stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

logger.info(
    "Training model 12, binary classification with default parameters and truncated feature set..."
)
model_12 = xgb.train(
    params_3,
    dtrain_binary_trunc,
    num_boost_round=2000,
    evals=watchlist_binary_trunc,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

logger.info(
    "Training model 13, binary classification with more regularization and tuned learning rate and truncated feature set..."
)
model_13 = xgb.train(
    params_4,
    dtrain_binary_trunc,
    num_boost_round=2000,
    evals=watchlist_binary_trunc,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds, stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)

logger.info(
    "Training model 14, binary classification with more regularization and tuned learning rate, scale_pos_weight and truncated feature set..."
)
model_14 = xgb.train(
    params_5,
    dtrain_binary_trunc,
    num_boost_round=2000,
    evals=watchlist_binary_trunc,
    # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds
    # stop training
    early_stopping_rounds=100,
    verbose_eval=100,
)


classification_models_and_data = [
    (
        model_9,
        dtrain_binary,
        dval_binary,
        "binary outcome, default parameters, early stopping",
    ),
    (
        model_10,
        dtrain_binary,
        dval_binary,
        "binary outcome, more regularization and tuned learning rate, early stopping",
    ),
    (
        model_11,
        dtrain_binary,
        dval_binary,
        "binary outcome, more regularization and tuned learning rate, scale_pos_weight, early stopping",
    ),
    (
        model_12,
        dtrain_binary_trunc,
        dval_binary_trunc,
        "binary outcome, truncated feature set, default parameters, early stopping",
    ),
    (
        model_13,
        dtrain_binary_trunc,
        dval_binary_trunc,
        "binary outcome, truncated feature set, more regularization and tuned learning rate, early stopping",
    ),
    (
        model_14,
        dtrain_binary_trunc,
        dval_binary_trunc,
        "binary outcome, truncated feature set, more regularization and tuned learning rate, scale_pos_weight, early stopping",
    ),
]

for max_update_num, (
    partial_premature_training_set,
    partial_premature_validation_set,
) in dtrain_dval_tuples_for_partial_premature_features.items():
    logger.info(
        "Training binary classification model with premature adaptive sandwich features up to update number %d...",
        max_update_num,
    )
    model = xgb.train(
        params_3,
        partial_premature_training_set,
        num_boost_round=2000,
        evals=[
            (partial_premature_training_set, "train"),
            (partial_premature_validation_set, "val"),
        ],
        # if validation metric (uses first non-training eval set) doesn't improve for 100 rounds
        # stop training
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    classification_models_and_data.append(
        (
            model,
            partial_premature_training_set,
            partial_premature_validation_set,
            f"binary outcome, premature adaptive sandwich features up to update number {max_update_num}, early stopping",
        )
    )

# ---------------------------------------------------------------------
# 6. Performance report for all models, examine feature importance,
#    then persist artifacts
# ---------------------------------------------------------------------
logger.info("Evaluating models...")

if args.regression_label_type == "log1p_trace":
    back_transformation_func = np.expm1
else:
    back_transformation_func = (
        lambda x: x
    )  # No transformation needed if we didn't take a log of the outputs

# Apply the back transformation to the validation labels (and the training
# label mean baseline predictor) if necessary
y_val = back_transformation_func(y_val)
y_bar = back_transformation_func(y_train.mean())

rmse_baseline = np.sqrt(((y_train - y_bar) ** 2).mean())
logger.info(
    "Baseline Model Validation RMSE (training-mean-only, raw scale): %.4f",
    rmse_baseline,
)

for i, (model, train, val, description) in enumerate(regression_models_and_data, 1):
    logger.info("Evaluating model %d: %s", i, description)
    if hasattr(model, "best_iteration"):
        # early stopping occurred: don't include trees
        # that hurt or at least didn't help validation performance
        val_pred = model.predict(val, iteration_range=(0, model.best_iteration + 1))
    else:
        # no early stopping: use the full model
        val_pred = model.predict(val)

    # Move the validation set predictions back to the raw scale if necessary
    val_pred = back_transformation_func(val_pred)

    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    logger.info("Model %d Validation RMSE (raw scale): %.4f", i, rmse)

    mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
    logger.info(
        "Model %d Validation Mean-absolute-percentage error (raw scale): %.2f%%",
        i,
        mape,
    )

    r2 = r2_score(y_val, val_pred)
    logger.info(
        "Model %d R^2 score between validation set and predictions (log scale): %.4f",
        i,
        r2,
    )

    plt.figure(figsize=(4, 4))
    plt.scatter(y_val, val_pred, s=8, alpha=0.6, linewidths=0)
    lims = [
        min(y_val.min(), val_pred.min()),
        max(y_val.max(), val_pred.max()),
    ]
    plt.plot(lims, lims, color="k", lw=1)
    plt.xlabel("True Adaptive Sandwich Trace")
    plt.ylabel("Predicted Trace")
    plt.title("Validation Set Parity plot (raw scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "parity_plot.png"), dpi=150)
    plt.close()

    # ---------- residual histogram ----------
    resid = y_val - val_pred
    plt.figure(figsize=(4, 3))
    plt.hist(resid, bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title("Residual histogram for Validation Set (raw scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_hist.png"), dpi=150)
    plt.close()

    # # Feature importance by weight (number of times feature used in a split)
    # importance = model.get_score(importance_type="weight")
    # importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    # logger.info("Model %d top 100 features (by weight):", i)
    # for feat, score in importance[:100]:
    #     logger.info("Model %d feature: %s, Weight: %s", i, feat, score)

    # Feature importance by total gain (total loss reduction of splits which use the feature)
    importance = model.get_score(importance_type="total_gain")
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Model %d top 100 features (by total gain):", i)
    for feat, score in importance[:100]:
        logger.info("Model %d feature: %s, Total Gain: %s", i, feat, score)

    model_path = os.path.join(OUTPUT_DIR, f"adaptive_sandwich_xgb_model_{i}.json")
    model.save_model(model_path)
    feature_order_path = os.path.join(OUTPUT_DIR, f"feature_order_model_{i}.txt")
    with open(feature_order_path, "w", encoding="utf-8") as f:
        for col in df.columns:
            f.write(col + "\n")
    logger.info("Model %d saved to %s", i, model_path)


# Calculate the mode of the training binary labels
majority_class = mode(y_train_binary, keepdims=True).mode[0]
binary_accuracy_baseline = (y_val_binary == majority_class).mean()


logger.info(
    "Baseline Validation Accuracy (always predict majority class): %.4f",
    binary_accuracy_baseline,
)
for i, (model, train, val, description) in enumerate(classification_models_and_data, 1):
    logger.info("Evaluating binary classification model %d: %s", i, description)
    if hasattr(model, "best_iteration"):
        # early stopping occurred: don't include trees
        # that hurt or at least didn't help validation performance
        val_pred = model.predict(val, iteration_range=(0, model.best_iteration + 1))
    else:
        # no early stopping: use the full model
        val_pred = model.predict(val)

    val_pred_binary = (val_pred > 0.5).astype("int32")
    accuracy = (val_pred_binary == y_val_binary).mean()
    logger.info("Model %d Validation Accuracy: %.4f", i, accuracy)

    precision = (y_val_binary[val_pred_binary == 1] == 1).mean()
    logger.info("Model %d Validation Precision: %.4f", i, precision)

    recall = (val_pred_binary[y_val_binary == 1] == 1).mean()
    logger.info("Model %d Validation Recall: %.4f", i, recall)

    f1 = f1_score(y_val_binary, val_pred_binary)
    logger.info("Model %d Validation F1 Score: %.4f", i, f1)

    # Plotting ROC curve
    fpr, tpr, _ = roc_curve(
        y_val_binary, val_pred
    )  # Get false positive rate, true positive rate, and thresholds
    roc_auc = auc(fpr, tpr)
    logger.info("Model %d Validation ROC AUC: %.4f", i, roc_auc)

    # Feature importance by total gain (total loss reduction of splits which use the feature)
    importance = model.get_score(importance_type="total_gain")
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Model %d top 100 features (by total gain):", i)
    for feat, score in importance[:100]:
        logger.info("Model %d feature: %s, Total Gain: %s", i, feat, score)

    model_path = os.path.join(
        OUTPUT_DIR, f"adaptive_sandwich_xgb_model_binary_{i}.json"
    )
    model.save_model(model_path)
    logger.info("Binary classification Model %d saved to %s", i, model_path)
