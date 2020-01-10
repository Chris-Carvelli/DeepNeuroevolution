import os
import torch

from src.ga import GA
from examples.atari.Controller import PolicyNN
from examples.atari.HyperNN import HyperNN

runs = {
    'hnn': lambda obs_space, action_space: HyperNN(obs_space, action_space, PolicyNN, 512),
    'ann_0.625': lambda obs_space, action_space: PolicyNN(obs_space, action_space, 0.125),
    'ann_0.125': lambda obs_space, action_space: PolicyNN(obs_space, action_space, 0.125),
    'ann_0.25': lambda obs_space, action_space: PolicyNN(obs_space, action_space, 0.25),
    'ann_0.5': lambda obs_space, action_space: PolicyNN(obs_space, action_space, 0.5),
    'ann': lambda obs_space, action_space: PolicyNN(obs_space, action_space),
}


def main(run):
    env = 'SkiingDeterministic-v4'
    max_evals = 60000000000
    ga = GA(
        env_key=env,
        population=1000,
        model_builder=runs[run],
        max_evals=max_evals,
        max_generations=1000,
        sigma=0.002,
        min_sigma=0.002,
        truncation=3,
        trials=1,
        elite_trials=20,
        n_elites=1,
        save_folder='results/skiing',
        run_name=run
    )

    res = True
    while res is not False:
        res = ga.optimize()


if __name__ == "__main__":
    for run in runs:
        main(run)

