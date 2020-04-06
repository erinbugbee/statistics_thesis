import numpy as np

class Policy:
    # Gives the policy the agent uses for action selection
    def choose(self, agent, num_arms):
        return 0


class GreedyPolicy(Policy):

    def __init__(self):
        return

    def choose(self, agent, n, trial): # Chooses by the greedy policy, picking the option with the highest Q-value
        Q = agent.get_Q()
        idx = np.argmax(Q, axis=0)
        choice = idx[trial]
        return choice


class EpsilonGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, agent, n, trial):
        # Chooses the option with the highest Q-value with prob 1 - epsilon, chooses randomly with prob epsilon
        if np.random.random() < self.epsilon:
            # choose random option
            return np.random.choice(n)
        else:
            # choose greedy option
            Q = agent.get_Q()
            idx = np.argmax(Q, axis=0)
            choice = idx[trial]
            return choice


class RandomPolicy(Policy):
    def choose(self, agent, n):
        return np.random.choice(n)


class SoftmaxEpsilonPolicy(Policy):
    def choose(self, agent, n, invT):
        Q = agent.get_Q()
        Q = Q - max(Q)
        pChoice = np.exp(Q * invT) / np.nansum(np.exp(Q * invT))
        choice = np.index(np.cumsum(pChoice) > np.randint(0, 1), 1)
        return choice
