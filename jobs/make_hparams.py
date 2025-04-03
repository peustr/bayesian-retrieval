import copy
import numpy as np
from collections import OrderedDict


def prod_dict(hyp_dict):
    # single element values
    fixed_values = {k: v for (k, v) in hyp_dict.items() if not isinstance(v, list)}

    # grid values
    varying_values = {k: v for (k, v) in hyp_dict.items() if k not in fixed_values}

    hyperparameters = OrderedDict(
        sorted(varying_values.items(), key=lambda _: _[0]))
    keys = list(hyperparameters.keys())

    indices = [len(values) for (arg, values) in hyperparameters.items()]
    choices = []
    for idx_choice in np.ndindex(*indices):
        # copy over the fixed values
        one_choice = copy.deepcopy(fixed_values)
        # pick current values
        for arg_idx, (arg, val_idx) in enumerate(zip(keys, idx_choice)):
            one_choice[arg] = hyperparameters[arg][val_idx]
        choices.append(one_choice)

    return choices


def make_cmd(params) -> str:
    cmd = ""
    for p, v in params.items():
        if isinstance(v, bool):
            # only add if it's True
            if v:
                cmd += f"--{p}  "
        else:
            cmd += f"--{p} {v} "

    return " " + cmd


def make_params(dest, param_dict, chkpt_format):
    with open(dest, 'w') as writer:
        n = 0
        for choice in prod_dict(param_dict):
            ckpt_file_name = chkpt_format.format(**choice)
            choice["ckpt_file_name"] = ckpt_file_name
            writer.write(make_cmd(choice) + "\n")
            n += 1
        print(f"wrote {n} params to {dest}")


if __name__ == '__main__':
    # DPR
    dpr_params = {
        "lr": [1e-3, 1e-4, 1e-5, 1e-6],
        "model_name": ["bert-base", "distilbert-base"]
    }

    make_params("jobs/hyperparams/dpr.params", dpr_params, "output/hs/dpr-{model_name}_{lr}.pt")

    # BRET params
    bret_params = {
        "lr": [1e-3, 1e-4, 1e-5, 1e-6],
        "model_name": ["bert-base", "distilbert-base"]
    }

    make_params("jobs/hyperparams/bret.params", bret_params, "output/hs/bret-{model_name}_{lr}.pt")

    # MC-Dropout params
    mcd_params = {
        "lr": [1e-3, 1e-4, 1e-5, 1e-6],
        "model_name": ["bert-base", "distilbert-base"]
    }

    make_params("jobs/hyperparams/mcdropout.params", mcd_params, "output/hs/mcdrop-{model_name}_{lr}.pt")
