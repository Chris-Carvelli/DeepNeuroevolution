import os
import torch
import click

from src.ga import GA
from examples.atari.Controller import PolicyNN
from examples.atari.HyperNN import HyperNN

# envs = [
#     'Amidar',
#     'Assault',
#     'Asterix'
#     'Atlantis',
#     'Enduro',
#     'Gravitar',
#     'Kangaroo',
#     'Seaquest',
#     'Zaxxon'
# ]

runs = {
    'ann': lambda obs_space, action_space: PolicyNN(obs_space, action_space),
    'hnn': lambda obs_space, action_space: HyperNN(obs_space, action_space, PolicyNN, 512),
    'ann_0.5': lambda obs_space, action_space: PolicyNN(obs_space, action_space, 0.5),
    'ann_0.25': lambda obs_space, action_space: PolicyNN(obs_space, action_space, 0.25),
    'ann_0.125': lambda obs_space, action_space: PolicyNN(obs_space, action_space, 0.125),
}


@click.command()
@click.argument('run')
@click.argument('env')
def main(run, env):
    max_evals = 1e9

    ga = GA(
        env_key=f'{env}Deterministic-v4',
        population=1000,
        model_builder=runs[run],
        max_evals=max_evals,
        max_episode_eval=5000,
        max_generations=200000,
        sigma=0.002,
        min_sigma=0.002,
        truncation=20,
        trials=1,
        elite_trials=5,
        n_elites=1,
        save_folder=f'results/{env}',
        run_name=run,
        reproduce_policy='2way'
    )

    res = True
    while res is not False:
        res = ga.optimize()


if __name__ == "__main__":
    main()

