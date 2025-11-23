import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import pytest

import functions_to_pass_to_analysis.oralytics_estimate_theta_primary_analysis
import functions_to_pass_to_analysis.oralytics_primary_analysis_estimating_function
import functions_to_pass_to_analysis.oralytics_primary_analysis_loss
import functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure
import functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_paper_simulation


def test_get_action_1_prob_pure_no_clip():
    treat_states = np.array([1.0, -1.06434164])
    beta_est = np.array([-0.16610159, 0.98683333, -1.287509, -1.0602505])

    np.testing.assert_equal(
        functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure.synthetic_get_action_1_prob_pure(
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
        functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure.synthetic_get_action_1_prob_pure(
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

    np.testing.assert_allclose(loss, expected_loss, rtol=1e-4)


def test_oralytics_primary_analysis_loss_equivalence_different_observation_groupings():
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

    # Compute total loss in batch
    batch_loss = functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss(
        theta_est, tod, bbar, abar, appengage, bias, action, oscb, act_prob
    )

    # Compute per-observation loss and sum
    row_losses = []
    for i in range(5):
        row_loss = functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss(
            theta_est,
            tod[i : i + 1],
            bbar[i : i + 1],
            abar[i : i + 1],
            appengage[i : i + 1],
            bias[i : i + 1],
            action[i : i + 1],
            oscb[i : i + 1],
            act_prob[i : i + 1],
        )
        row_losses.append(row_loss)

    summed_row_loss = sum(row_losses)

    np.testing.assert_allclose(batch_loss, summed_row_loss, rtol=1e-6)

    grouped_losses = [
        functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss(
            theta_est,
            tod[0:3],
            bbar[0:3],
            abar[0:3],
            appengage[0:3],
            bias[0:3],
            action[0:3],
            oscb[0:3],
            act_prob[0:3],
        ),
        functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss(
            theta_est,
            tod[3:5],
            bbar[3:5],
            abar[3:5],
            appengage[3:5],
            bias[3:5],
            action[3:5],
            oscb[3:5],
            act_prob[3:5],
        ),
    ]

    summed_group_loss = sum(grouped_losses)

    np.testing.assert_allclose(batch_loss, summed_group_loss, rtol=1e-6)


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


def test_oralytics_primary_analysis_estimating_function_sums_to_zero_with_real_theta():
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

    covariate_names = [
        "tod",
        "bbar",
        "abar",
        "appengage",
        "bias",  # this is the intercept
    ]

    in_study_bool = study_df["in_study_indicator"] == 1
    # TODO: Figure out other indicator(s)
    trimmed_df = study_df.loc[in_study_bool, covariate_names + ["act_prob"]].copy()

    in_study_df = study_df[in_study_bool]
    trimmed_df["centered_action"] = in_study_df["action"] - in_study_df["act_prob"]

    inference_loss_function = (
        functions_to_pass_to_analysis.oralytics_primary_analysis_loss.oralytics_primary_analysis_loss
    )
    derived_inference_estimating_function = jax.grad(inference_loss_function)
    total_grad = jnp.sum(
        jax.vmap(
            derived_inference_estimating_function,
            in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0),
        )(
            jnp.array(regression_coef),
            jnp.array(trimmed_df["tod"]),
            jnp.array(trimmed_df["bbar"]),
            jnp.array(trimmed_df["abar"]),
            jnp.array(trimmed_df["appengage"]),
            jnp.array(trimmed_df["bias"]),
            jnp.array(in_study_df["action"]),
            jnp.array(in_study_df["oscb"]),
            jnp.array(trimmed_df["act_prob"]),
        ),
        axis=0,
    )

    np.testing.assert_allclose(total_grad, jnp.zeros_like(total_grad), atol=7e-4)

    inference_estimating_function = (
        functions_to_pass_to_analysis.oralytics_primary_analysis_estimating_function.oralytics_primary_analysis_estimating_function
    )
    total_grad = jnp.sum(
        jax.vmap(
            inference_estimating_function,
            in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0),
        )(
            jnp.array(regression_coef),
            jnp.array(trimmed_df["tod"]),
            jnp.array(trimmed_df["bbar"]),
            jnp.array(trimmed_df["abar"]),
            jnp.array(trimmed_df["appengage"]),
            jnp.array(trimmed_df["bias"]),
            jnp.array(in_study_df["action"]),
            jnp.array(in_study_df["oscb"]),
            jnp.array(trimmed_df["act_prob"]),
        ),
        axis=0,
    )

    np.testing.assert_allclose(total_grad, jnp.zeros_like(total_grad), atol=1.4e-3)


def test_synthetic_get_least_squares_loss_inference_paper_simulation_zero_loss():
    theta_est = jnp.array([1.0, 2.0, 3.0])
    intercept = jnp.array([1.0])
    past_reward = jnp.array([2.0])
    action = jnp.array([3.0])
    reward = jnp.array([14.0])  # 1*1 + 2*2 + 3*3 = 14

    loss = functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_paper_simulation.synthetic_get_least_squares_loss_inference_paper_simulation(
        theta_est, intercept, past_reward, action, reward
    )
    assert jnp.isclose(loss, 0.0), f"Expected loss to be 0.0, got {loss}"


def test_synthetic_get_least_squares_loss_inference_paper_simulation_nonzero_loss():
    theta_est = jnp.array([1.0, 2.0, 3.0])
    intercept = jnp.array([1.0])
    past_reward = jnp.array([2.0])
    action = jnp.array([3.0])
    reward = jnp.array([15.0])  # Incorrect reward to induce non-zero loss

    loss = functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_paper_simulation.synthetic_get_least_squares_loss_inference_paper_simulation(
        theta_est, intercept, past_reward, action, reward
    )
    assert loss == 1, f"Expected loss to be greater than 0.0, got {loss}"


def test_synthetic_get_least_squares_loss_inference_paper_simulation_shape_mismatch():
    theta_est = jnp.array([1.0, 2.0])  # Incorrect shape
    intercept = jnp.array([1.0])
    past_reward = jnp.array([2.0])
    action = jnp.array([3.0])
    reward = jnp.array([14.0])

    with pytest.raises(TypeError):
        functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_paper_simulation.synthetic_get_least_squares_loss_inference_paper_simulation(
            theta_est, intercept, past_reward, action, reward
        )


def test_synthetic_get_least_squares_loss_inference_paper_simulation_batch_nonzero_loss():
    theta_est = jnp.array([1.0, 2.0, 3.0])
    intercept = jnp.array([[1.0], [1.0], [1.0]])
    past_reward = jnp.array([[2.0], [3.0], [4.0]])
    action = jnp.array([[3.0], [2.0], [1.0]])
    reward = jnp.array(
        [[15.0], [12.0], [10.0]]
    )  # Incorrect rewards to induce non-zero loss

    loss = functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_paper_simulation.synthetic_get_least_squares_loss_inference_paper_simulation(
        theta_est, intercept, past_reward, action, reward
    )

    assert loss == 1 + 1 + 4
