import gym
from gym import wrappers
import torch
import pickle

from examples.racing.models.worldmodels.PolicyNN import PolicyNN
from examples.racing.models.HyperNN import HyperNN

run = 'results/racing/hnn-world-models'
env_key = 'CarRacing-v0'

env = wrappers.Monitor(gym.make(env_key), 'plots/videos/racing/hnn-world-models/')
# controller = PolicyNN(env.observation_space, env.action_space)
# controller = HyperNN(env.observation_space, env.action_space, PolicyNN, 1024)
# controller.load_state_dict(torch.load(f'{run}/best.p'))
with open(f'{run}/best.p', 'rb') as fp:
    controller = pickle.load(fp)[0]

print(controller.evaluate(env, -1, True))
env.close()
