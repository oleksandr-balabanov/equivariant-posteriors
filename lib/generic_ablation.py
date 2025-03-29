import itertools

from lib.train_distributed import request_train_run


def generic_ablation(create_config, values_dict):
    combinations = itertools.product(*values_dict.values())
    kwarg_names = list(values_dict.keys())
    configs = []
    for values in combinations:
        kwargs = {name: val for name, val in zip(kwarg_names, values)}
        train_run = create_config(**kwargs)
        configs.append(train_run)
        request_train_run(train_run)

    return configs
