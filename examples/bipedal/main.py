import gym

from src.ga import GA
from examples.bipedal.Controller import PolicyNN


def main():
    env = gym.make('BipedalWalker-v2')
    max_evals = 60000000000
    ga = GA(
        env_key=env,
        population=200,
        model_builder=lambda obs_space, action_space:  PolicyNN(obs_space, action_space),
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
        res = ga.optimize()


if __name__ == "__main__":
    main()
