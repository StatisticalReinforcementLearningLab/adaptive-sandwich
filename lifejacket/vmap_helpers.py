import collections

import numpy as np
from jax import numpy as jnp


def get_shape(obj):
    if hasattr(obj, "shape"):
        return obj.shape
    if isinstance(obj, str):
        return None
    try:
        return len(obj)
    except Exception:
        return None


def group_user_args_by_shape(user_arg_dict, empty_allowed=True):
    user_arg_dicts_by_shape = collections.defaultdict(dict)
    for user_id, args in user_arg_dict.items():
        if not args:
            if not empty_allowed:
                raise ValueError("There shouldn't be a user with no data at this time")
            continue
        shape_id = tuple(get_shape(arg) for arg in args)
        user_arg_dicts_by_shape[shape_id][user_id] = args
    return user_arg_dicts_by_shape.values()


def stack_batched_arg_lists_into_tensors(batched_arg_lists):
    """
    Stack a simple Python list of lists of function arguments into a list of jnp arrays that can be
    supplied to vmap as batch arguments. vmap requires all elements of such a batched array to be
    the same shape, as do the stacking functions we use here.  Thus we require this be called on
    batches with the same data shape. We also supply the axes one must iterate over to get
    each users's args in a batch.

    Every failure raised here names the offending argument position, because the
    underlying jnp.stack/vstack/array errors do not -- and for user-supplied
    argument tuples, input_checks.require_supplied_arg_types_supported rejects
    unsupported entry types before any of this runs, with the key and subject
    id attached.
    """

    batched_arg_tensors = []

    # This ends up being all zeros because of the way we are (now) doing the
    # stacking, but better to not assume that externally and send out what
    # we've done with this list.
    batch_axes = []

    for position, batched_arg_list in enumerate(batched_arg_lists):
        if isinstance(batched_arg_list, (jnp.ndarray, np.ndarray)):
            # Already a single (bucket_size, ...) tensor -- e.g. a
            # jax.jit-traced array reconstructed by
            # post_deployment_analysis._rebuild_bucket_from_jit_arrays from a
            # genuine jit argument, rather than a plain Python list of
            # per-subject values. Use it as-is instead of round-tripping
            # through list(...)/jnp.stack below, which would otherwise add
            # one slice + one restack graph op per subject in the bucket for
            # a tensor that is already exactly the shape/axis-0-batched form
            # this function exists to produce. This is unreachable for every
            # existing caller (build_batched_arg_lists_by_subject always
            # supplies a plain Python list), so it changes no existing
            # behavior. (jnp.asarray on an already-jnp array/tracer is an
            # identity-preserving no-op -- see the passthrough unit test's
            # `is` assertion -- and moves a numpy input to device.)
            if batched_arg_list.ndim == 0:
                # A 0-D array has no axis-0 batch dimension to map over --
                # accepting it here would only fail later, inside jax.vmap,
                # with a confusing non-local error. No current caller can
                # produce this (see above), but this function has a history
                # of 0-D misclassification via isinstance-only dispatch
                # (see the NOTE below), so fail loudly and locally instead.
                raise TypeError(
                    f"Argument position {position}: expected an already-stacked "
                    "(bucket_size, ...) array, got a 0-D (scalar-shaped) array. "
                    "A scalar-per-subject argument position must be supplied "
                    "as a plain Python list of per-subject scalars instead."
                )
            batched_arg_tensors.append(jnp.asarray(batched_arg_list))
            batch_axes.append(0)
            continue
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
            raise TypeError(
                f"Argument position {position}: arrays with more than 2 "
                "dimensions are not supported."
            )
        if isinstance(first, (jnp.ndarray, np.ndarray)) and first.ndim == 2:
            ########## We have a matrix (2D array) arg
            try:
                stacked = jnp.stack(batched_arg_list, 0)
            except Exception as e:
                # Blame the actual cause: a ValueError with shape advice only when
                # the shapes really do differ, a TypeError otherwise (e.g. an
                # object- or datetime-dtype array jnp cannot convert) -- shape
                # advice for a dtype problem would send the caller the wrong way.
                if len({getattr(x, "shape", None) for x in batched_arg_list}) > 1:
                    raise ValueError(
                        f"Argument position {position}: could not stack the "
                        f"per-subject 2-D arrays into one batch tensor ({e}). "
                        "Every subject in a batch must supply the same array "
                        "shape at each argument position."
                    ) from e
                raise TypeError(
                    f"Argument position {position}: the per-subject 2-D arrays "
                    f"cannot be converted to a JAX numpy batch tensor ({e}). "
                    "Arrays must have a numeric or boolean dtype."
                ) from e
            batched_arg_tensors.append(stacked)
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
                        f"Argument position {position}: sequence-type argument "
                        "that cannot be cast to a JAX numpy array. Sequences "
                        "must contain only numbers."
                    ) from e
                if batched_arg_list[0].ndim != 1:
                    raise TypeError(
                        f"Argument position {position}: a sequence-type argument "
                        "must be FLAT (one number per entry); got one that casts "
                        f"to a {batched_arg_list[0].ndim}-D array. Supply nested "
                        "data as a 2-D array instead."
                    )
            try:
                stacked = jnp.vstack(batched_arg_list)
            except Exception as e:
                # Same cause-splitting as the 2-D branch above.
                if len({getattr(x, "shape", None) for x in batched_arg_list}) > 1:
                    raise ValueError(
                        f"Argument position {position}: could not stack the "
                        f"per-subject 1-D vectors into one batch tensor ({e}). "
                        "Every subject in a batch must supply the same vector "
                        "length at each argument position."
                    ) from e
                raise TypeError(
                    f"Argument position {position}: the per-subject 1-D vectors "
                    f"cannot be converted to a JAX numpy batch tensor ({e}). "
                    "Arrays must have a numeric or boolean dtype."
                ) from e
            batched_arg_tensors.append(stacked)
            batch_axes.append(0)
        else:
            ########## Otherwise we should have a list of scalars (plain
            # Python ints/floats, or 0-D arrays/tracers). Just turn into a
            # jnp array.
            try:
                batched_arg_tensors.append(jnp.array(batched_arg_list))
            except Exception as e:
                raise TypeError(
                    f"Argument position {position}: expected one scalar per "
                    f"subject, got entries like {type(first).__name__!r} that "
                    "cannot be cast to a JAX numpy array. Supported argument "
                    "kinds are numeric scalars, flat numeric sequences, and "
                    "1-D or 2-D numeric arrays."
                ) from e
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
    (calculate_derivatives.get_batched_arg_lists_and_involved_user_ids, now
    legacy/debug-only, uses inspect.signature -- which handles jax wrappers
    but still over-counts when the function declares defaulted parameters the
    caller does not supply).
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
