import time

import gym
import torch
import torch.nn as nn
from torch.autograd import Variable


class PolicyNN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(24, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
            nn.Tanh(),
        )

        self.add_tensors = {}
        self.init()

    def forward(self, x):
        x = self.nn(x)
        x = x[0]

        return x

    def evolve(self, sigma):
        params = self.named_parameters()
        for name, tensor in sorted(params):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)

            tensor.data.add_(to_add)

    def init(self):
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())

    def evaluate(self, env_key, max_eval, render=False, fps=60):
        if isinstance(env_key, str):
            env = gym.make(env_key, verbose=0)
            should_close = True
        else:
            env = env_key
            should_close = False

        self.eval()

        tot_reward = 0
        neg_count = 0
        n_eval = 0
        is_done = False
        max_eval = max_eval or -1

        while not is_done:
            # removed some scaffolding, check if something was needed
            values = self(Variable(torch.Tensor([state])))
            # values = env.step(env.action_space.sample())
            action = values.detach().numpy()

            state, reward, done, _ = env.step(action)
            if render:
                env.render('human')
                time.sleep(1/fps)

            tot_reward += reward
            n_eval += 1

        if should_close:
            env.close()
            gc.collect()
        return tot_reward, n_eval
