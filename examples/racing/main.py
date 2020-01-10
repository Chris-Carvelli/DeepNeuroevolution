import os
import torch

from src.ga import GA
from examples.racing.models.worldmodels.PolicyNN import PolicyNN
from examples.racing.models.HyperNN import HyperNN

torch.set_num_threads(1)

print(os.getcwd())

runs = {
    # 'ann_0.125': lambda obs_space, action_space: PolicyNN(obs_space, action_space, shrink=0.125),
    'hnn_0.125': lambda obs_space, action_space: HyperNN(obs_space, action_space, PolicyNN, 1024, 0.125),
    # 'hnn-128': lambda obs_space, action_space: HyperNN(obs_space, action_space, PolicyNN, 128),
}


def main(run):
    max_evals = 60000000000
    ga_pol = GA(
        env_key='CarRacing-v0',
        population=200,
        model_builder=runs[run],
        max_evals=max_evals,
        max_generations=1000,
        sigma=0.01,
        min_sigma=0.01,
        truncation=3,
        trials=1,
        elite_trials=20,
        n_elites=1,
        save_folder='results/racing',
        run_name=run
    )

    res = True
    while res is not False:
        res = ga_pol.optimize()


if __name__ == "__main__":
    for run in runs:
        main(run)
