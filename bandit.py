import numpy as np


class MultiArmedBandit:

    def __init__(self, n):
        self.n = n
        self.is_best = True # Whether the action is the best action (has the highest value)

    def reset_values(self):
        pass

    def pull_arm(self, action):
        pass


class GaussianBandit:
    def __init__(self, n, mu=0, sigma=1, obs_sigma=3):
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.obs_sigma = obs_sigma
        self.reset_values()
        self.action_values = None
        self.best_action = None

    def reset_values(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.n)
        self.best_action = np.argmax(self.action_values)

    def pull_arm(self, action):
        reward = np.random.normal(self.action_values[action], self.obs_sigma)

        if action == self.best_action:
            self.is_best = True
        else:
            self.is_best = False

        return reward, self.is_best
