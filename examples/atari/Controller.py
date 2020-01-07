import random
import time
import gc

import numpy as np
import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

IMG_SIZE = 64

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor()
])


class PolicyNN(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNN, self).__init__()

        self.state_dim = obs_space.shape[0]
        self.action_dim = action_space.n

        self.vision = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), 1),
            nn.ReLU()
        )

        self.controller = nn.Sequential(
            nn.Linear(4 * 4 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 18),
            nn.Softmax()
        )

        # self.conv1 = nn.Conv2d(4, 32, (8, 8), 4)
        # self.conv2 = nn.Conv2d(32, 64, (4, 4), 2)
        # self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)
        # self.dense = nn.Linear(4 * 4 * 64, 512)
        # self.out = nn.Linear(512, 18)

        self.add_tensors = {}
        self.init()

    def forward(self, x: torch.Tensor):
        x = self.vision(x)
        x = x.view(1, -1)
        x = self.controller(x)

        return x.squeeze().numpy()

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
        with torch.no_grad():
            if isinstance(env_key, str):
                env = gym.make(env_key)
                should_close = True
            else:
                env = env_key
                should_close = False
            state = env.reset()
            cur_states = transform(state).repeat(4, 1, 1) / 255.0
            env.render('rgb_array')

            self.eval()

            tot_reward = 0
            neg_count = 0
            n_eval = 0
            is_done = False
            max_eval = max_eval or -1

            for _ in range(random.randint(0, 30)):
                cur_states = cur_states[1:]
                new_state, reward, is_done, _ = env.step(0)
                new_state = transform(new_state) / 255.0
                tot_reward += reward
                if is_done:
                    return tot_reward
                cur_states = torch.cat((cur_states, new_state))

            while not is_done:
                values = self(cur_states.unsqueeze(0))
                action = np.argmax(values[:env.action_space.n])

                state, reward, done, _ = env.step(action)
                state = transform(state)
                cur_states = cur_states[1:]
                cur_states = torch.cat((cur_states, state))

                # neg_count = neg_count + 1 if reward <= 0.0 else 0
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
