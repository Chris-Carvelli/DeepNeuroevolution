import gym

from src.ga import GA
from examples.bipedal.Controller import PolicyNN


def main():
    ga = GA(
            env=gym.make('BipedalWalker-v2'),
            population=10,
            model_builder=lambda: PolicyNN(),
            max_evals=6000000,
            max_generations=50,
            sigma=0.1,
            truncation=2,
            trials=1,
            elite_trials=0,
            n_elites=1,
            graphical_output=False
            )

    res = True
    while res is not False:
        res = ga.optimize()
        print(res)


if __name__ == "__main__":
    main()
