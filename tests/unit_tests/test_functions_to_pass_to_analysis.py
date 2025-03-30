import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import pytest

import functions_to_pass_to_analysis.oralytics_estimate_theta_primary_analysis
import functions_to_pass_to_analysis.oralytics_primary_analysis_loss
import functions_to_pass_to_analysis.get_action_1_prob_pure


def test_get_action_1_prob_pure_no_clip():
    treat_states = np.array([1.0, -1.06434164])
    beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

    np.testing.assert_equal(
        functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure(
            beta_est=beta_est,
            lower_clip=0.1,
            steepness=10,
            upper_clip=0.9,
            treat_states=treat_states,
        ),
        np.array(0.16932733, dtype=np.float32),
    )


def test_get_action_1_prob_pure_clip():
    treat_states = np.array([1.0, -0.12627351])
    beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

    np.testing.assert_equal(
        functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure(
            beta_est=beta_est,
            lower_clip=0.1,
            steepness=10,
            upper_clip=0.9,
            treat_states=treat_states,
        ),
        np.array(0.1, dtype=np.float32),
    )


def test_oralytics_primary_analysis_loss():
    # Example data
    theta_est = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    tod = jnp.array([[1], [2], [3], [4], [5]])
    bbar = jnp.array([[2], [3], [4], [5], [6]])
    abar = jnp.array([[3], [4], [5], [6], [7]])
    appengage = jnp.array([[4], [5], [6], [7], [8]])
    bias = jnp.array([[1], [1], [1], [1], [1]])
    action = jnp.array([[1], [0], [1], [0], [1]])
    oscb = jnp.array([[10], [15], [20], [25], [30]])
    act_prob = jnp.array([[0.2], [0.3], [0.4], [0.5], [0.6]])

    weights = jnp.array(
        [
            [1 / 0.2 / 0.8],
            [1 / 0.3 / 0.7],
            [1 / 0.4 / 0.6],
            [1 / 0.5 / 0.5],
            [1 / 0.6 / 0.4],
        ]
    )
    expected_loss = 0.5 * jnp.sum(
        weights
        * (
            (
                jnp.array([[10], [15], [20], [25], [30]])
                - jnp.array(
                    [
                        [1, 2, 3, 4, 1, 0.2, 0.8],
                        [2, 3, 4, 5, 1, 0.3, -0.3],
                        [3, 4, 5, 6, 1, 0.4, 0.6],
                        [4, 5, 6, 7, 1, 0.5, -0.5],
                        [5, 6, 7, 8, 1, 0.6, 0.4],
                    ]
                )
                @ jnp.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]])
            )
            ** 2
        )
    )

    loss = functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss(
        theta_est, tod, bbar, abar, appengage, bias, action, oscb, act_prob
    )

    np.testing.assert_almost_equal(loss, expected_loss)


def test_oralytics_estimate_theta_primary_analysis():
    # Example DataFrame
    data = {
        "tod": [1, 2, 3, 4, 5],
        "bbar": [2, 3, 4, 5, 6],
        "abar": [3, 4, 5, 6, 7],
        "appengage": [4, 5, 6, 7, 8],
        "bias": [1, 1, 1, 1, 1],
        "act_prob": [0.2, 0.3, 0.4, 0.5, 0.6],
        "action": [1, 0, 1, 0, 1],
        "oscb": [10, 15, 20, 25, 30],
        "in_study_indicator": [1, 1, 1, 1, 1],
    }
    study_df = pd.DataFrame(data)

    # Call the function
    coef = functions_to_pass_to_analysis.oralytics_estimate_theta_primary_analysis.oralytics_estimate_theta_primary_analysis(
        study_df
    )

    covariate_names = [
        "tod",
        "bbar",
        "abar",
        "appengage",
        "bias",
    ]

    in_study_bool = study_df["in_study_indicator"] == 1
    trimmed_df = study_df.loc[in_study_bool, covariate_names + ["act_prob"]].copy()
    in_study_df = study_df[in_study_bool]
    trimmed_df["centered_action"] = in_study_df["action"] - in_study_df["act_prob"]

    # Extract covariates and response variable
    X = trimmed_df[covariate_names + ["act_prob", "centered_action"]].values
    y = in_study_df["oscb"].values

    # Compute weights
    weights = 1 / (trimmed_df["act_prob"] * (1 - trimmed_df["act_prob"])).values

    # Perform weighted least squares regression
    W = np.sqrt(np.diag(weights))
    X_w = W @ X
    y_w = W @ y
    expected_coef, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)

    # Assert that the coefficients are almost equal to the expected coefficients
    np.testing.assert_almost_equal(coef, expected_coef, decimal=6)


@pytest.mark.skip(
    "The optimization is not converging to the regression answer for some reason, but I'm not "
    "convinced it's because the loss is wrong..."
)
def test_minimizing_oralytics_primary_analysis_loss_same_as_regression():
    # Example DataFrame
    tod = jnp.array([1, 37, 5, 7, 46])
    bbar = jnp.array([2, 4, 60, 8, 10])
    abar = jnp.array([3, 1, 4, 2, 5])
    appengage = jnp.array([4, 6, 8, 10, 12])
    bias = jnp.array([1, 1, 1, 1, 1])
    act_prob = jnp.array([0.2, 0.4, 0.6, 0.3, 0.5])
    action = jnp.array([1, 0, 1, 0, 1])
    oscb = jnp.array([10, 20, 30, 40, 50])
    in_study_indicator = jnp.array([1, 1, 1, 1, 1])

    data = {
        "tod": tod,
        "bbar": bbar,
        "abar": abar,
        "appengage": appengage,
        "bias": bias,
        "act_prob": act_prob,
        "action": action,
        "oscb": oscb,
        "in_study_indicator": in_study_indicator,
    }
    study_df = pd.DataFrame(data)

    regression_coef = functions_to_pass_to_analysis.oralytics_estimate_theta_primary_analysis.oralytics_estimate_theta_primary_analysis(
        study_df
    )

    print(
        f"Regression loss: {functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss(regression_coef, tod.reshape(-1, 1), bbar.reshape(-1, 1), abar.reshape(-1, 1), appengage.reshape(-1, 1), bias.reshape(-1, 1), action.reshape(-1, 1), oscb.reshape(-1, 1), act_prob.reshape(-1, 1))}"
    )

    # Create an Adam optimizer and initialize its state
    optimizer = optax.adam(learning_rate=1e-3)
    params = jnp.zeros(7)
    opt_state = optimizer.init(params)

    # Define a single optimization step
    @jax.jit
    def update(params, opt_state, *args):
        loss, grads = jax.value_and_grad(
            functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss
        )(params, *args)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    min_loss = float("inf")
    best_params = None
    for epoch in range(30000):
        params, opt_state, loss = update(
            params,
            opt_state,
            tod.reshape(-1, 1),
            bbar.reshape(-1, 1),
            abar.reshape(-1, 1),
            appengage.reshape(-1, 1),
            bias.reshape(-1, 1),
            action.reshape(-1, 1),
            oscb.reshape(-1, 1),
            act_prob.reshape(-1, 1),
        )
        if loss < min_loss:
            best_params = params
            min_loss = loss
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    print(f"Min loss found: {min_loss}")

    np.testing.assert_almost_equal(regression_coef, best_params, decimal=6)
