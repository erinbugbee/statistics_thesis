import numpy as np


class Agent:
    # The agent is the decision maker. They choose actions and update their estimates according to Q-learning.
    def __init__(self, n, policy, num_trials):
        self.n = n #the number of options
        self.policy = policy
        self.num_trials = num_trials
        self.Q = np.zeros((self.n, self.num_trials)) # The Q-values
        self.num_attempts = np.zeros(self.n) # The number of times each option has been chosen
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
        self.num_attempts[self.prev_action] += 1 # Increment the number of attempts for the chosen action

        for a in range(self.n):
            if a == chosen_action:
                if self.opt_LR: # Time-varying learning rate
                    self.learning_rate = (1 / self.num_attempts[a])
                else: # Fixed learning rate
                    self.learning_rate = self.fixed_LR_vec
                # Update Q-value for that action at the next trial according to Q-learning
                self.Q[a, trial + 1] = self.Q[a, trial] + self.learning_rate * (reward - self.Q[a, trial])
            else:
                self.Q[a, trial + 1] = self.Q[a, trial]
