import numpy as np


class Agent:
    def __init__(self, n, policy, num_trials):
        self.n = n
        self.policy = policy
        self.num_trials = num_trials
        self.Q = np.zeros((self.n, self.num_trials))
        self.num_attempts = np.zeros(self.n)
        self.prev_action = None
        self.opt_LR = True
        self.learning_rate = None
        self.fixed_LR_vec = None

    def choose_action(self, trial):
        action = self.policy.choose(self, self.n, trial)
        self.prev_action = action
        return action

    def reset(self):
        self.Q = np.zeros((self.n, self.num_trials))
        self.num_attempts = np.zeros(self.n)

    def get_Q(self):
        return self.Q

    def update_estimates(self, trial, chosen_action, reward):
        self.num_attempts[self.prev_action] += 1

        for a in range(self.n):
            if a == chosen_action:
                if self.opt_LR:
                    self.learning_rate = (1 / self.num_attempts[a])
                else:
                    self.learning_rate = self.fixed_LR_vec
                self.Q[a, trial + 1] = self.Q[a, trial] + self.learning_rate * (reward - self.Q[a, trial])
            else:
                self.Q[a, trial + 1] = self.Q[a, trial]
