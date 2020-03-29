from environment import Environment
from bandit import GaussianBandit
from agent import Agent
from policy import (EpsilonGreedyPolicy, GreedyPolicy, SoftmaxEpsilonPolicy)
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})
import numpy as np


def compare_epsilons(epsilons):
    # maximizer: epsilon = 1, complete exploration
    # satisficer: epsilon = 0, complete exploitation
    rewards = np.zeros((len(epsilons), num_sessions, num_trials))
    num_best = np.zeros((len(epsilons), num_sessions, num_trials))

    for i in range(len(epsilons)):
        policy = EpsilonGreedyPolicy(epsilons[i])
        bandit = GaussianBandit(n)
        agent = Agent(n, policy, num_trials)
        env = Environment(bandit, agent, num_trials, num_sessions)
        rewards[i, :, :], num_best[i, :, :] = env.run()

    # Compare average reward across values of epsilon
    for i in range(len(epsilons)):
        ave_reward = rewards[i, :, :].mean(axis=0)
        plt.plot(ave_reward, label="Epsilon:" + str(epsilons[i]))
        plt.title("Average Reward")
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")
    plt.show()

    for i in range(len(epsilons)):
        ave_percent_best = num_best[i, :, :].mean(axis=0)
        plt.plot(ave_percent_best, label="Epsilon:" + str(epsilons[i]))
        plt.title("Average Percent Best Option")
        plt.xlabel('Trial')
        plt.ylabel('Percent Best Option')
        plt.legend(loc="upper left")
    plt.show()

def greedy_policy():
    rewards = np.zeros((len(epsilons), num_sessions, num_trials))
    num_best = np.zeros((len(epsilons), num_sessions, num_trials))

    for i in range(len(epsilons)):
        policy = GreedyPolicy()
        bandit = GaussianBandit(n)
        agent = Agent(n, policy, num_trials)
        env = Environment(bandit, agent, num_trials, num_sessions)
        rewards[i, :, :], num_best[i, :, :] = env.run()

    ave_reward = rewards[i, :, :].mean(axis=0)
    plt.plot(ave_reward)
    plt.title("Average Reward")
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.show()

    ave_percent_best = num_best[i, :, :].mean(axis=0)
    plt.plot(ave_percent_best)
    plt.title("Average Percent Best Option")
    plt.xlabel('Trial')
    plt.ylabel('Percent Best Option')
    plt.show()


def compare_n(n_list):

    rewards = np.zeros((len(n_list), num_sessions, num_trials))
    num_best = np.zeros((len(n_list), num_sessions, num_trials))

    for i in range(len(n_list)):
        policy = EpsilonGreedyPolicy(epsilon)
        bandit = GaussianBandit(n)
        agent = Agent(n, policy, num_trials)
        env = Environment(bandit, agent, num_trials, num_sessions)
        rewards[i, :, :], num_best[i, :, :] = env.run()

    # Compare average reward across values of epsilon
    for i in range(len(n_list)):
        ave_reward = rewards[i, :, :].mean(axis=0)
        plt.plot(ave_reward, label="n:" + str(n_list[i]))
        plt.title("Average Reward")
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")
    plt.show()

    for i in range(len(n_list)):
        ave_percent_best = num_best[i, :, :].mean(axis=0)
        plt.plot(ave_percent_best, label="n:" + str(n_list[i]))
        plt.title("Average Percent Best Option")
        plt.xlabel('Trial')
        plt.ylabel('Percent Best Option')
        plt.legend(loc="upper left")
    plt.show()


# Loss by not picking the optimal action
def plot_regret():
    # average regret versus trials averaged over sessions
    ave_regrets = regrets.mean(axis=0)
    plt.plot(ave_regrets)
    plt.title("Average Regrets")
    pass


def plot_ave_reward(rewards):
    # average reward versus trials averaged over sessions
    ave_reward = rewards.mean(axis=0)
    plt.plot(ave_reward)
    plt.title("Average Reward")
    plt.xlabel('Trial')
    plt.ylabel('Reward')

    return ave_reward


def plot_percent_best_action(num_best):
    # percent best action versus trials averaged over sessions for each time step
    ave_percent_best = num_best.mean(axis=0)
    plt.plot(ave_percent_best)
    plt.title("Average Percent Best Option")
    plt.xlabel('Trial')
    plt.ylabel('Percent Best Option')


def run_bandit(epsilon, num_trials, num_sessions):
    # rewards = np.zeros((len(epsilon), num_sessions, num_trials))
    policy = EpsilonGreedyPolicy(epsilon)
    bandit = GaussianBandit(n)
    agent = Agent(n, policy, num_trials)
    env = Environment(bandit, agent, num_trials, num_sessions)
    rewards, num_best = env.run()

    plot_ave_reward(rewards)
    plt.show()

    plot_percent_best_action(num_best)
    plt.show()


if __name__ == '__main__':
    num_trials = 500
    num_sessions = 2000
    n = 10
    epsilon = 0.1

    # COMPARE EPSILONS FOR EPSILON GREEDY
    #epsilons = np.array((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    # epsilons = np.array((0, 0.2, 0.4, 0.6, 0.8, 1))
    epsilons = np.array((0, 1))
    #compare_epsilons(epsilons)

    # COMPARE N FOR EPSILON GREEDY
    #n_list = np.array((2, 5, 10, 20))
    #compare_n(n_list)

    #run_bandit(epsilon, num_trials, num_sessions)

    #greedy_policy()
