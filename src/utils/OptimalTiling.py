from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import gym
import seaborn as sns


env_key = 'CarRacing-v0'

if env_key == 'CarRacing-v0':
    from examples.racing.models.worldmodels.PolicyNN import PolicyNN
    from examples.racing.models.HyperNN import HyperNN
    tile_sizes = [1024]
else:
    from examples.atari.Controller import PolicyNN
    from examples.atari.HyperNN import HyperNN
    tile_sizes = [512]

env = gym.make(env_key)
df = pd.DataFrame()


def count_params(state_dict):
    ret = 0
    for param in state_dict.parameters():
        ret += reduce(lambda x, y: x * y, param.shape, 1)

    return ret


def count_buffers(state_dict):
    ret = 0
    for buffer in state_dict.buffers():
        ret += reduce(lambda x, y: x * y, buffer.shape, 1)
    return ret


def count_trainable(state_dict, filter=None):
    ret = 0
    for name in state_dict:
        if filter is None:
            ret += reduce(lambda x, y: x * y, state_dict[name].shape, 1)
        elif filter in name:
            ret += reduce(lambda x, y: x * y, state_dict[name].shape, 1)
    return ret


def main():
    shrink_factors = [1, 0.75, 0.5, 0.25, 0.125]
    for shrink in shrink_factors:
        pnn = PolicyNN(env.observation_space, env.action_space, shrink=shrink)
        df = df.append(
            {
                'model': f'pnn_{shrink}',
                'z_size': 0,
                'params_size': count_params(pnn),
                'kind': 'main',
            },
            ignore_index=True
        )

    for tile_size in tile_sizes:
        hnn = HyperNN(env.observation_space, env.action_space, PolicyNN, tile_size)
        df = df.append(
            {
                'model': f'hnn{tile_size}',
                'z_size': count_buffers(hnn),
                'params_size': count_params(hnn.hnn),
                'kind': 'main',
            },
            ignore_index=True
        )

    df['tot'] = df['z_size'] + df['params_size']
    print(df)
    palette = (
        sns.color_palette('Oranges_r', len(shrink_factors)) +
        sns.color_palette('Blues_r', 1)
    )
    # ax = df[['model', 'z_size', 'params_size']].plot(kind='bar', stacked=True)
    # ax.set_xticklabels(df['model'], rotation=45)

    ax0 = plt.bar(df['model'], df['params_size'], label='Weights/biases', color=sns.color_palette('Purples', 1))
    ax1 = plt.bar(df['model'], df['z_size'], bottom=df['params_size'], label='Z vector', color=sns.color_palette('Greens', 1))

    plt.xticks(df['model'])
    plt.title(env_key)
    plt.legend()
    plt.show()
    plt.savefig('param_size.png')
    df.to_csv('param_size.csv')
