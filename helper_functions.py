import numpy as np
import jax
from jax import numpy as jnp


def conditional_x_or_one_minus_x(x, condition):
    return (1 - condition) + (2 * condition - 1) * x


def clip(args, vals):
    lower_clipped = np.maximum(vals, args.lower_clip)
    clipped = np.minimum(lower_clipped, args.upper_clip)
    return clipped


def get_utri(matrix, return_idx=False):
    triu_idx = np.triu_indices(matrix.shape[0])
    if return_idx:
        return triu_idx
    return matrix[triu_idx]


def symmetric_fill_utri(values, mdim):
    matrix = np.zeros((mdim, mdim))
    triu_idx = np.triu_indices(mdim)
    matrix[triu_idx] = values
    return matrix + matrix.T - np.diag(np.diag(matrix))


def removeBinaryDuplicate(varname, feat_list):
    if varname in feat_list:
        tmplist = [x for x in feat_list if x != varname]
        if varname == "intercept" and len(set(feat_list)) == 1:
            # only add "intercept" variable in if there are no other features
            tmplist.append(varname)
        elif varname != "intercept":
            tmplist.append(varname)
        tmplist.sort()
        return tmplist
    else:
        return feat_list


def alg2varnames(RLalg):
    all_feats = RLalg.state_feats + ["action," + x for x in RLalg.treat_feats]

    binary_vars = ["intercept", "action"]

    feat_matrix_names = []
    state_feats = []
    for feat1 in all_feats:
        # form state features
        feat1list = feat1.split(",")
        feat1list.sort()
        state_feats.append(",".join(feat1list))

        for feat2 in all_feats:
            tmp_feat_list = feat1.split(",") + feat2.split(",")
            tmp_feat_list.sort()

            for featname in binary_vars:
                tmp_feat_list = removeBinaryDuplicate(featname, tmp_feat_list)

            feat_matrix_names.append(",".join(tmp_feat_list))

    suffvec_names = set(feat_matrix_names)
    suffvec_names.remove("intercept")
    suffvec_names = list(suffvec_names)
    suffvec_names.sort()

    return {
        "flat_matrix_names": feat_matrix_names,
        "suffvec_names": suffvec_names,
        "state_feats": state_feats,
    }


def var2suffvec(RLalg, varmatrix, return_idx=False):
    # TODO: Add documentation about how this is constructed
    # grabbing unique values but order is determined by feature list somewhere?
    # Should we include intercept term or not?
    if RLalg.action_centering:
        varmatrix_flat = get_utri(varmatrix)

        if return_idx:
            mdim = varmatrix.shape[0]
            idx_matrix = np.arange(0, mdim * mdim).reshape(mdim, mdim)
            all_idx = get_utri(idx_matrix)
            return all_idx[1:]

        # remove intercept term
        return varmatrix_flat[1:]

    else:
        var_name_dict = alg2varnames(RLalg)
        flat_matrix_names = var_name_dict["flat_matrix_names"]
        suffvec_names = var_name_dict["suffvec_names"]

        all_idx = []
        for suff_feat in suffvec_names:
            idx = flat_matrix_names.index(suff_feat)
            all_idx.append(idx)

        if return_idx:
            return all_idx
        return varmatrix.flatten()[all_idx]


def suffvec2var(RLalg, suffvec, intercept_val):
    if RLalg.action_centering:
        suffvec_full = np.concatenate([[intercept_val], suffvec])
        varmatrix = symmetric_fill_utri(suffvec_full, RLalg.prior_mean.shape[0])

    else:
        var_name_dict = alg2varnames(RLalg)
        flat_matrix_names = var_name_dict["flat_matrix_names"]
        suffvec_names = var_name_dict["suffvec_names"]

        flat_matrix_vals = []
        for name in flat_matrix_names:
            if name == "intercept":
                flat_matrix_vals.append(intercept_val)
            else:
                tmp_idx = suffvec_names.index(name)
                flat_matrix_vals.append(suffvec[tmp_idx])

        dim = RLalg.prior_mean.shape[0]
        varmatrix = np.array(flat_matrix_vals).reshape(dim, dim)

    # Check that variance matrix is symmetric
    assert np.allclose(varmatrix, varmatrix.T)

    return varmatrix
