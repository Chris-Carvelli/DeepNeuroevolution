import torch
import gym
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

from examples.atari.Controller import PolicyNN
from examples.atari.HyperNN import HyperNN

# params
env_key = 'Gravitar'
run = 'ann'

# computed
tmp = run.split('_')
nn_key, shrink = (run, 1) if len(tmp) == 1 else tmp
folder = f'results/{env_key}/{run}'

env = gym.make(f'{env_key}Deterministic-v4')
if nn_key == 'hnn':
    nn = HyperNN(env.observation_space, env.action_space, PolicyNN, 512, shrink)
else:
    nn = PolicyNN(env.observation_space, env.action_space, shrink)

state_dict = torch.load(f'{folder}/best.p')
nn.load_state_dict(state_dict)


policy_nn, w_plt_folder = (nn.pnn, 'indirect') if nn_key is 'hnn' else (nn, 'direct')
for name, param in policy_nn.named_parameters():
    if 'bias' not in name:
        data = param.detach().numpy()
        if len(np.shape(data)) == 4:
            data = np.reshape(data, (data.shape[0] * data.shape[2], data.shape[1] * data.shape[3]))
        plt.figure(figsize=(data.shape[1]/32, data.shape[0]/32))
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        # plt.title(name)
        plt.savefig(f'plots/weights/{w_plt_folder}/{name}.png', bbox_inches='tight')
        # plt.show()

# z_shard = nn.z_indexer[name]
# sns.heatmap(nn.z_v[z_shard], square=True, cmap='gray')
