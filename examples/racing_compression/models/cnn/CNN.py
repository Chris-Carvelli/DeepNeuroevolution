import random
import numpy as np
import math
import gc
import gym
import time
from itertools import chain
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

# Hardcoded for now. Note: Size of latent vector (LSIZE) is increased to 128 for DISCRETE representation
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64
models = ['vision', 'controller']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


class PolicyNN(nn.Module):
    def __init__(self, obs_space, action_space):
        n = obs_space.shape[0]
        m = obs_space.shape[1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        super(PolicyNN, self).__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2),
            nn.ReLU()
        )

        self.controller = nn.Sequential(
            nn.Linear(384, action_space.shape[0])
        )

        self.add_tensors = {}
        self.init()

    def forward(self, obs):
        x = self.vision(obs)
        x = x.view(x.size(0), -1)
        action = self.controller(x)

        return action.squeeze().cpu().numpy()

        # x = torch.transpose(torch.transpose(x, 1, 3), 2, 3)
        # x = self.vision(x)
        # x = x.reshape(x.shape[0], -1)
        # x = self.controller(x)
        # return x

    def evolve(self, sigma):
        params = self.named_parameters(recurse=True)
        coin_toss = math.floor(random.random() * len(models))

        module_filter = models[coin_toss]

        for name, tensor in sorted(params):
            if module_filter in name:
                to_add = self.add_tensors[tensor.size()]
                to_add.normal_(0.0, sigma)

                tensor.data.add_(to_add)

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())

    def evaluate(self, env_key, max_eval, render=False, fps=60, early_termination=True):
        with torch.no_grad():
            if isinstance(env_key, str):
                env = gym.make(env_key, verbose=0)
                should_close = True
            else:
                env = env_key
                should_close = False
            state = env.reset()
            env.render('rgb_array')

            self.eval()

            tot_reward = 0
            neg_count = 0
            n_eval = 0
            is_done = False
            max_eval = max_eval or -1

            while not is_done:
                state = transform(state).unsqueeze(0)

                action = self(state)

                state, reward, is_done, _ = env.step(action)
                # Count how many times the car did not get a reward (e.g. was outside track)
                neg_count = neg_count + 1 if reward < 0.0 else 0
                # To speed up training, determinte evaluations that are outside of track too many times
                if neg_count > 20 and early_termination:
                    is_done = True

                if render:
                    # print(f'render eval {n_eval}')
                    # print('action=%s, reward=%.2f' % (action, reward))
                    env.render('human')
                    time.sleep(1/fps)

                tot_reward += reward
                n_eval += 1
                if -1 < max_eval <= n_eval:
                    break

            if should_close:
                env.close()
                gc.collect()
            return tot_reward, n_eval
