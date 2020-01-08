from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import gym

from examples.racing_compression.models.cnn.CNN import PolicyNN
from examples.racing.models.cnn.CNN import PolicyNN as MainNN
from examples.racing_compression.models.HyperNN import HyperNN

env = gym.make('CarRacing-v0')
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

for i in range(9, 10):
    hnn = HyperNN(env.observation_space, env.action_space, MainNN, 2**i)
    df = df.append(
        {
            'model': f'hnn{2**i}',
            'z_size': count_buffers(hnn),
            'params_size': count_params(hnn.hnn)
        },
        ignore_index=True
    )

df['tot'] = df['z_size'] + df['params_size']
print(df)
ax = df[['model', 'z_size', 'params_size']].plot(kind='bar', stacked=True)
# plt.show()
plt.savefig('param_size.png')
df.to_csv('param_size.csv')
