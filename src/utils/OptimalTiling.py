from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import gym

from examples.atari.Controller import PolicyNN
from examples.atari.HyperNN import HyperNN

env = gym.make('FrostbiteDeterministic-v4')
df = pd.DataFrame()


def count_params(nn):
    ret = 0
    for param in nn.parameters():
        print(param.shape)
        ret += reduce(lambda x, y: x * y, param.shape, 1)
    print(ret)

    return ret


def count_buffers(nn):
    ret = 0
    for buffer in nn.buffers():
        print(buffer.shape)
        ret += reduce(lambda x, y: x * y, buffer.shape, 1)
    print(ret)
    return ret


pnn = PolicyNN(env.observation_space, env.action_space)
df = df.append(
    {
        'model': 'pnn',
        'z_size': 0,
        'params_size': count_params(pnn)
    },
    ignore_index=True
)

for i in range(3, 15):
    hnn = HyperNN(env.observation_space, env.action_space, PolicyNN, 2**i)
    df = df.append(
        {
            'model': f'hnn{2**i}',
            'z_size': count_buffers(hnn),
            'params_size': count_params(hnn.hnn)
        },
        ignore_index=True
    )

print(df)
ax = df[['model', 'z_size', 'params_size']].plot(kind='bar', stacked=True)
plt.show()
df.to_csv('param_size.csv')
