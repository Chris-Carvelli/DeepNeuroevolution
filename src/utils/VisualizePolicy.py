import click
import tqdm
from statistics import mean, stdev, median
import gym
import torch
from examples.atari.Controller import PolicyNN
from examples.atari.HyperNN import HyperNN

run_to_env = {
    'atari': 'FrostbiteDeterministic-v4',
    'skiing': 'SkiingNoFrameskip-v4',
    'venture': 'VentureDeterministic-v4',
    'asteroids': 'AsteroidsNoFrameskip-v4',
}


@click.command()
@click.option(
    '--run',
    type=click.Path(),
    required=True,
    help='Path to run folder'
)
@click.option(
    '--random_policy',
    is_flag=True,
    help='Test random policy',
)
@click.option(
    '--render',
    is_flag=True,
    help='Render environment',
)
@click.option(
    '--n',
    default=1,
    help='Number of runs',
)
def main(run, random_policy, render, n):
    env_key = run.split('/')[1]
    if env_key in run_to_env:
        env_key = run_to_env[env_key]
    else:
        env_key = f'{env_key}Deterministic-v4'
    shrink = run.split('_')[-1]
    model = run.split('/')[-1].split('_')[0]

    try:
        shrink = float(shrink)
    except ValueError:
        shrink = 1

    # if save_folder is not None:
    #     env = wrappers.Monitor(gym.make(env_key), save_folder, force=overwrite_video)
    # else:
    env = gym.make(env_key)

    if model == 'ann':
        controller = PolicyNN(env.observation_space, env.action_space, shrink)
    else:
        controller = HyperNN(env.observation_space, env.action_space, PolicyNN, 512)

    if not random_policy:
        state_dict = torch.load(f'{run}/best.p')
        controller.load_state_dict(state_dict)

    res = []
    for _ in tqdm.tqdm(range(n)):
        res.append(controller.evaluate(env_key, -1, render))
    env.close()

    if len(res) == 1:
        print(res[0])
    else:
        scores = list(map(lambda x: x[0], res))
        evals = list(map(lambda x: x[1], res))
        print(f'min: {min(scores)} ({min(evals)})')
        print(f'max: {max(scores)} ({max(evals)})')
        print(f'avg: {mean(scores)} ({mean(evals)})')
        print(f'std: {stdev(scores)} ({stdev(evals)})')
        print(f'media: {median(scores)} ({median(evals)})')

    return scores


scores = None
if __name__ == '__main__':
    scores = main()
