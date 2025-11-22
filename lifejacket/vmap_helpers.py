import collections

from jax import numpy as jnp
import numpy as np


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
        if (
            isinstance(
                batched_arg_list[0],
                (jnp.ndarray, np.ndarray),
            )
            and batched_arg_list[0].ndim > 2
        ):
            raise TypeError("Arrays with dimension greater that 2 are not supported.")
        if (
            isinstance(
                batched_arg_list[0],
                (jnp.ndarray, np.ndarray),
            )
            and batched_arg_list[0].ndim == 2
        ):
            ########## We have a matrix (2D array) arg

            batched_arg_tensors.append(jnp.stack(batched_arg_list, 0))
            batch_axes.append(0)
        elif isinstance(
            batched_arg_list[0],
            (collections.abc.Sequence, jnp.ndarray, np.ndarray),
        ) and not isinstance(batched_arg_list[0], str):
            ########## We have a vector (1D array) arg
            if not isinstance(batched_arg_list[0], (jnp.ndarray, np.ndarray)):
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
            ########## Otherwise we should have a list of scalars.
            # Just turn into a jnp array.
            batched_arg_tensors.append(jnp.array(batched_arg_list))
            batch_axes.append(0)

    return (
        batched_arg_tensors,
        batch_axes,
    )
