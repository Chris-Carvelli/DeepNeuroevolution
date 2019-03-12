import time

import torch
import torch.nn as nn
from torch.autograd import Variable


class PolicyNN(nn.Module):
    def __init__(self):
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

    def evaluate(self, env, max_eval, render=False, fps=60):
        state = env.reset()

        self.eval()

        tot_reward = 0
        n_eval = 0
        done = False
        while not done:
            # removed some scaffolding, check if something was needed
            values = self(Variable(torch.Tensor([state])))
            # values = env.step(env.action_space.sample())
            action = values.detach().numpy()

            state, reward, done, _ = env.step(action)
            if render:
                env.render()
                time.sleep(1/fps)

            tot_reward += reward
            n_eval += 1

        return tot_reward, n_eval
