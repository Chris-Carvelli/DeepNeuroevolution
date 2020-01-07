import time
import gc

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PolicyNN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNN, self).__init__()

        print(action_space.shape)
        self.state_dim = obs_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.max_action = float(action_space.high[0])

        # self.nn = nn.Sequential(
        #     nn.Linear(self.state_dim, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, self.action_dim),
        #     nn.Tanh(),
        # )

        self.l1 = nn.Linear(self.state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, self.action_dim)

        self.add_tensors = {}
        self.init()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x[0]
        # x = self.nn(x)
        # x = x[0] * self.max_action
        #
        # return x

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

    def evaluate(self, env_key, max_eval, render=False, fps=60, early_termination=True):
        if isinstance(env_key, str):
            env = gym.make(env_key)
            should_close = True
        else:
            env = env_key
            should_close = False
        state = env.reset()

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

            state, reward, is_done, _ = env.step(action)
            neg_count = neg_count + 1 if reward < 0.0 else 0
            # if neg_count > 20 and early_termination:
            #     is_done = True

            if render:
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
