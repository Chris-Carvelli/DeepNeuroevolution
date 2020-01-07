from src.ga import GA
from examples.box2d.Controller import PolicyNN
from examples.box2d.HyperNN import HyperNN

runs = {
    'ann': lambda obs_space, action_space: PolicyNN(obs_space, action_space),
    'hnn': lambda obs_space, action_space: HyperNN(obs_space, action_space, PolicyNN),
}


def main(run):
    env = 'LunarLanderContinuous-v2'
    max_evals = 60000000000
    ga = GA(
        env_key=env,
        population=1000,
        model_builder=lambda obs_space, action_space: PolicyNN(obs_space, action_space),
        max_evals=max_evals,
        max_generations=1000,
        sigma=0.001,
        min_sigma=0.001,
        truncation=20,
        trials=1,
        elite_trials=30,
        n_elites=1,
        save_folder='results/lander',
        run_name=run
    )

    res = True
    while res is not False:
        res = ga.optimize()


if __name__ == "__main__":
    for run in runs:
        main(run)
