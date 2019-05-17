import gym
import gym_minigrid
from src.ga import GA
from examples.minigrid.HyperNN import HyperNN


def main():
    ga = GA(
        gym.make('MiniGrid-Empty-5x5-v0'),
        population=200,
        model_builder=lambda obs_space, action_space: HyperNN(obs_space, action_space),
        max_evals=6000000,
        max_generations=100,
        sigma=0.05,
        truncation=10,
        trials=1,
        elite_trials=0,
        n_elites=1,
    )

    res = True
    while res is not False:
        res = ga.optimize()


if __name__ == "__main__":
    main()
