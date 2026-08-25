import collections

import numpy as np
from jax import numpy as jnp


# TODO: Check for exactly the required types earlier
# TODO: Try except and nice error message
# TODO: This is complicated enough to deserve its own unit tests
def stack_batched_arg_lists_into_tensors(batched_arg_lists):
    """
    Stack a simple Python list of lists of function arguments into a list of jnp arrays that can be
    supplied to vmap as batch arguments. vmap requires all elements of such a batched array to be
    the same shape, as do the stacking functions we use here.  Thus we require this be called on
    batches with the same data shape. We also supply the axes one must iterate over to get
    each users's args in a batch.
    """

    batched_arg_tensors = []

    # This ends up being all zeros because of the way we are (now) doing the
    # stacking, but better to not assume that externally and send out what
    # we've done with this list.
    batch_axes = []

    for batched_arg_list in batched_arg_lists:
        first = batched_arg_list[0]
        # NOTE: isinstance(first, (jnp.ndarray, np.ndarray)) is True for a 0-D
        # (scalar-shaped) array too -- including a jax.jit-traced value, since
        # a plain Python scalar argument becomes a 0-D tracer once it crosses
        # a jax.jit boundary, even though outside of any trace it would never
        # satisfy this isinstance check at all. So ndim must be checked
        # explicitly for each case below (0-D falls through to the final
        # "list of scalars" branch) rather than assuming "isinstance array,
        # not 2-D" implies 1-D.
        if isinstance(first, (jnp.ndarray, np.ndarray)) and first.ndim > 2:
            raise TypeError("Arrays with dimension greater that 2 are not supported.")
        if isinstance(first, (jnp.ndarray, np.ndarray)) and first.ndim == 2:
            ########## We have a matrix (2D array) arg

            batched_arg_tensors.append(jnp.stack(batched_arg_list, 0))
            batch_axes.append(0)
        elif (isinstance(first, (jnp.ndarray, np.ndarray)) and first.ndim == 1) or (
            isinstance(first, collections.abc.Sequence) and not isinstance(first, str)
        ):
            ########## We have a vector (1D array, or plain sequence) arg
            if not isinstance(first, (jnp.ndarray, np.ndarray)):
                try:
                    batched_arg_list = [jnp.array(x) for x in batched_arg_list]
                except Exception as e:
                    raise TypeError(
                        "Argument of sequence type that cannot be cast to JAX numpy array"
                    ) from e
            assert batched_arg_list[0].ndim == 1

            batched_arg_tensors.append(jnp.vstack(batched_arg_list))
            batch_axes.append(0)
        else:
            ########## Otherwise we should have a list of scalars (plain
            # Python ints/floats, or 0-D arrays/tracers). Just turn into a
            # jnp array.
            batched_arg_tensors.append(jnp.array(batched_arg_list))
            batch_axes.append(0)

    return (
        batched_arg_tensors,
        batch_axes,
    )


def build_batched_arg_lists_by_subject(
    group_subject_ids: list[collections.abc.Hashable],
    args_by_subject_id: dict[collections.abc.Hashable, tuple],
) -> list[list]:
    """
    Stack a dict of {subject_id: args_tuple} (all sharing the same arg count,
    e.g. one group_user_args_by_shape bucket) into a list of per-position
    Python lists, in the exact subject order given.

    Deliberately derives the argument count from the data itself
    (len(args_by_subject_id[...])) rather than function introspection
    (calculate_derivatives.get_batched_arg_lists_and_involved_user_ids uses
    func.__code__.co_argcount, which is correct for a raw, undecorated
    function but wrong for a jax.grad(...)-wrapped one -- its __code__
    reflects the wrapper's own *args-style signature, not the wrapped
    function's).
    """
    num_args = len(args_by_subject_id[group_subject_ids[0]])
    return [
        [args_by_subject_id[subject_id][idx] for subject_id in group_subject_ids]
        for idx in range(num_args)
    ]


def batch_args_by_subject(
    group_subject_ids: list[collections.abc.Hashable],
    args_by_subject_id: dict[collections.abc.Hashable, tuple],
) -> tuple[list, list[int]]:
    return stack_batched_arg_lists_into_tensors(
        build_batched_arg_lists_by_subject(group_subject_ids, args_by_subject_id)
    )
