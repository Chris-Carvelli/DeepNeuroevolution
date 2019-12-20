import torch

from src.ga import GA
from examples.racing.models.PolicyNN import PolicyNN
from examples.racing.models.HyperNN import HyperNN

torch.set_num_threads(1)


def main():
    max_evals = 60000000000
    ga_pol = GA(
            env_key='CarRacing-v0',
            population=200,
            model_builder=lambda obs_space, action_space: HyperNN(obs_space, action_space, PolicyNN, 1024),
            max_evals=max_evals,
            max_generations=1000,
            sigma=0.01,
            min_sigma=0.01,
            truncation=3,
            trials=1,
            elite_trials=20,
            n_elites=1
            )

    res = True
    while res is not False:
        res = ga_pol.optimize()


if __name__ == "__main__":
    main()
