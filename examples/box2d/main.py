from src.ga import GA
from examples.box2d.Controller import PolicyNN
from examples.box2d.HyperNN import HyperNN

runs = {
    'ann_100': lambda obs_space, action_space: PolicyNN(obs_space, action_space, [40, 30]),
    'hnn_100': lambda obs_space, action_space: HyperNN(obs_space, action_space, PolicyNN, 8, 4, 8),
}


def main(run):
    env = 'LunarLanderContinuous-v2'
    max_evals = 60000000000
    ga = GA(
        env_key=env,
        population=200,
        model_builder=runs[run],
        max_evals=max_evals,
        max_generations=200,
        sigma=0.1,
        min_sigma=0.002,
        sigma_decay=0.99,
        truncation=3,
        trials=1,
        elite_trials=20,
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
