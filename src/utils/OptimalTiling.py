from functools import reduce
import pandas as pd
import gym

from examples.racing.models.cnn.CNN import CNN
from examples.racing.models.HyperNN import HyperNN

env = gym.make('CarRacing-v0')
df = pd.DataFrame()


def count_params(nn):
    ret0 = ret1 = 0
    for param in nn.parameters():
        print(param.shape)
        ret0 += reduce(lambda x, y: x * y, param.shape, 1)
    print(ret0)
    for buffer in nn.buffers():
        print(buffer.shape)
        ret1 += reduce(lambda x, y: x * y, buffer.shape, 1)
    print(ret1)
    return ret0,  ret1


pnn = CNN(env.observation_space, env.action_space)
p, z = count_params(pnn)
df = df.append(
    {
        'model': 'pnn',
        'z_size': z,
        'params_size': p
    },
    ignore_index=True
)

for i in range(5, 15):
    hnn = HyperNN(env.observation_space, env.action_space, CNN, 2**i)
    p, z = count_params(hnn)
    df = df.append(
        {
            'model': f'hnn{2**i}',
            'z_size': z,
            'params_size': p
        },
        ignore_index=True
    )

print(df)
df[['model', 'z_size', 'params_size']].plot(kind='bar', stacked=True)
df.to_csv('param_size.csv')
