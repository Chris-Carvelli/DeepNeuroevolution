import gym


class RolloutGenerator:
    def __init__(self, env_key, max_episode_eval=None, verbose=0):
        self.env_key = env_key
        self.max_episode_eval = max_episode_eval

        self.env = gym.make(env_key, verbose)

    def evaluate(self, env_key, render=False, fps=60, early_termination=True):
        state = self.env.reset()

