import numpy as np


class Environment:
    def __init__(self, bandit, agent, num_trials, num_sessions):
        self.bandit = bandit
        self.agent = agent
        self.num_trials = num_trials
        self.num_sessions = num_sessions
        self.rewards = np.zeros((self.num_sessions, self.num_trials))
        self.num_best = np.zeros((self.num_sessions, self.num_trials))

    def reset(self):
        self.bandit.reset_values()
        self.agent.reset()

    def run(self):
        for ses in range(self.num_sessions):
            self.reset()
            for trial in range(self.num_trials):
                chosen_action = self.agent.choose_action(trial)
                reward, self.is_best = self.bandit.pull_arm(chosen_action)
                self.rewards[ses, trial] = reward
                if trial != self.num_trials - 1:
                    self.agent.update_estimates(trial, chosen_action, reward)
                if self.is_best:
                    self.num_best[ses, trial] += 1

        return self.rewards, self.num_best
