import gym
import gym_minigrid
from src.ga import GA
from examples.minigrid.HyperNN import HyperNN


def main():
    env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
    ga = GA(
        env,
        population=500,
        model_builder=lambda obs_space, action_space: HyperNN(obs_space, action_space),
        max_evals=6000000,
        max_generations=100,
        sigma=0.005,
        truncation=10,
        trials=3,
        elite_trials=10,
        n_elites=1,
    )

    res = True
    g = 0
    while res is not False:
        res = ga.optimize()
        if g % 5 == 0:
            res[4][0][0].evaluate(env, 100, True)
        g += 1


if __name__ == "__main__":
    main()
