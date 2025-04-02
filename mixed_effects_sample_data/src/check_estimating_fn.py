import pickle as pkl
from LME_estimating_function import LME_estimating_function

PATH = r"C:\Users\susob\Research\LME_afterstudy\results\num_users100_num_time_steps10_seed0_delta_seed0_beta_mean[1]_beta_std[[1]]_gamma_std[[0.1]]_sigma_e20.1_policy_typemixed_effects\estimating_equation_function_dict.pkl"


def main():
    est_eq_fn_dict = pkl.load(open(PATH, "rb"))
    for key in est_eq_fn_dict.keys():
        print(key)
        sum_ = None
        for user in sorted(est_eq_fn_dict[key].keys()):
            if sum_ is None:
                sum_ = LME_estimating_function(*est_eq_fn_dict[key][user])
            else:
                sum_ += LME_estimating_function(*est_eq_fn_dict[key][user])
        print(sum_)


if __name__ == "__main__":
    main()
