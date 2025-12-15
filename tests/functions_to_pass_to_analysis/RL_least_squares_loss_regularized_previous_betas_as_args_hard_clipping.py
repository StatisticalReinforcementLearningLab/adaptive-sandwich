import jax
from jax import numpy as jnp
from jax import core
from jax.experimental import host_callback as hcb
import numpy as np

from functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure import (
    synthetic_get_action_1_prob_pure,
)


@jax.jit
def RL_least_squares_loss_regularized_previous_betas_as_args_hard_clipping(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    real_action1probs,
    pre_update_action1probs,
    previous_post_update_betas,
    # gives policy nums for each post-update decision time for which
    # arguments are being provided.
    post_update_policy_nums,
    lower_clip,
    steepness,
    upper_clip,
    action_centering,
    lambda_,
    n,
):

    beta_0_est = beta_est[: base_states.shape[1]].reshape(-1, 1)
    beta_1_est = beta_est[base_states.shape[1] :].reshape(-1, 1)

    # TODO: Use vmap to speed this up if necessary.
    action1probs = jnp.concatenate(
        [
            pre_update_action1probs,
            jnp.array(
                [
                    synthetic_get_action_1_prob_pure(
                        # note the policy_num - 2 since post update betas
                        # correspond to the policies starting from the first
                        # *post-update* policy FOR THIS USER. Note how this
                        # works for incremental recruitment as well.
                        # The zeroth index gives the initial policy, index
                        # 1 gives the first post-update policy, etc. But in
                        # this setting policy 1 is the first policy.  So policy 2
                        # corresponds to index 0, policy 3 to index 1, etc.
                        previous_post_update_betas[policy_num - 2].flatten(),
                        lower_clip,
                        steepness,
                        upper_clip,
                        # treat states start from the first decision time,
                        # limit to post update to align
                        treat_states[i + pre_update_action1probs.shape[0]],
                    )
                    for i, policy_num in enumerate(post_update_policy_nums)
                ]
            ).reshape(-1, 1),
        ]
    )

    def _host_check(args, transforms):
        """
        hcb is jax.experimental.host_callback.

        Host callbacks allow invoking Python/host-side code (with side effects)
        from JAX-traced/JIT code. Here we use hcb to run a check on the host and
        raise a normal Python exception if real_action1probs != action1probs,
        which cannot be raised directly inside JIT/traced execution.
        """
        (
            real,
            calc,
            post_update_policy_nums,
            pre_update_action1probs,
            previous_post_update_betas,
        ) = args
        if not np.allclose(real, calc):
            breakpoint()
            raise ValueError("real_action1probs does not match action1probs")
        return 0

    if isinstance(real_action1probs, core.Tracer) or isinstance(
        action1probs, core.Tracer
    ):
        # inside jitted/traced execution -> run host callback to perform the check and raise on host
        _ = hcb.id_tap(
            _host_check,
            (
                real_action1probs,
                action1probs,
                post_update_policy_nums,
                pre_update_action1probs,
                previous_post_update_betas,
            ),
        )
    else:
        # eager execution -> regular Python exception
        if not jnp.allclose(real_action1probs, action1probs):
            breakpoint()
            raise ValueError("real_action1probs does not match action1probs")

    actions = jnp.where(
        action_centering, actions.astype(jnp.float32) - action1probs, actions
    )
    return (
        jnp.einsum(
            "ij->",
            (
                rewards
                - jnp.einsum("ij,jk->ik", base_states, beta_0_est)
                - jnp.einsum("ij,jk->ik", actions * treat_states, beta_1_est)
            )
            ** 2,
        )
        + jnp.dot(
            beta_est,
            beta_est,
        )
        * lambda_
        / n
    )
