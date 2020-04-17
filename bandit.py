import numpy as np


class MultiArmedBandit:
    # The bandit is the "slot machine", which has n options
    def __init__(self, n):
        self.n = n
        self.is_best = True # Whether the action is the best action (has the highest value)

    def reset_values(self):
        pass

    def pull_arm(self, action):
        pass


class GaussianBandit:
    # Rewards are drawn from a Gaussian distribution when an arm is pulled
    def __init__(self, n, mu=0, sigma=1, obs_sigma=3):
        self.n = n
        self.mu = mu
        self.sigma = sigma # Standard deviation of true values
        self.obs_sigma = obs_sigma # Standard deviation of observed rewards
        self.reset_values()
        self.action_values = None
        self.best_action = None # To store the option with the highest mean value

    def reset_values(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.n) # Generates n normally distributed values
        self.best_action = np.argmax(self.action_values) # The best action has the highest action value

    def pull_arm(self, action):
        # Observe the reward sampled from the Gaussian distribution
        reward = np.random.normal(self.action_values[action], self.obs_sigma)

        # Check if the action is the best action
        if action == self.best_action:
            self.is_best = True
        else:
            self.is_best = False

        return reward, self.is_best
