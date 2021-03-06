import numpy as np
import random
from environment import Environment
from bandit import GaussianBandit
from agent import Agent
from policy import (EpsilonGreedyPolicy, GreedyPolicy, SoftmaxEpsilonPolicy)
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
plt.rcParams.update({'font.size': 18})


def compare_epsilons(n, epsilons):
    # Compare various values of n and epsilon

    # maximizer: epsilon = 1, complete exploration
    # satisficer: epsilon = 0, complete exploitation
    rewards = np.zeros((len(epsilons), num_sessions, num_trials))
    num_best = np.zeros((len(epsilons), num_sessions, num_trials))
    ave_reward = np.zeros((len(epsilons), num_trials))
    cum_reward = np.zeros(num_sessions)
    ave_cum_reward = np.zeros((len(epsilons), 2))

    for i in range(len(epsilons)):
        policy = EpsilonGreedyPolicy(epsilons[i])
        bandit = GaussianBandit(n)
        agent = Agent(n, policy, num_trials)
        env = Environment(bandit, agent, num_trials, num_sessions)
        rewards[i, :, :], num_best[i, :, :] = env.run()

    # Compare average reward across values of epsilon
    color = iter(cm.rainbow(np.linspace(0, 1, len(epsilons))))
    for i in range(len(epsilons)):
        c = next(color)
        ave_reward[i, :] = rewards[i, :, :].mean(axis=0)
        plt.plot(ave_reward[i, :], label="Epsilon:" + str(epsilons[i]), c=c)
        plt.title("Average Reward" + ", n: " + str(n))
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")
        plt.rc('legend', fontsize='x-small')
    plt.show()

    color2 = iter(cm.rainbow(np.linspace(0, 1, len(epsilons))))
    for i in range(len(epsilons)):
        c = next(color2)
        ave_percent_best = num_best[i, :, :].mean(axis=0)
        plt.plot(ave_percent_best, label="Epsilon:" + str(epsilons[i]), c=c)
        plt.title("Average Percent Best Option" + ", n: " + str(n))
        plt.xlabel('Trial')
        plt.ylabel('Percent Best Option')
        plt.legend(loc="upper left")
        plt.rc('legend', fontsize='x-small')
    plt.show()

    for i in range(len(epsilons)):
        for j in range(num_sessions):
            cum_reward[j] = rewards[i, j, :].sum()
        ave_cum_reward[i, :] = [epsilons[i], np.mean(cum_reward)]
    print(np.shape(cum_reward))
    print(np.shape(ave_cum_reward))
    print(ave_cum_reward)

def greedy_policy():
    # Agent chooses by the greedy policy
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
    # Compare across values of n
    rewards = np.zeros((len(n_list), num_sessions, num_trials))
    num_best = np.zeros((len(n_list), num_sessions, num_trials))
    cum_reward = np.zeros(num_sessions)
    ave_cum_reward = np.zeros((len(n_list), 2))

    for i in range(len(n_list)):
        policy = EpsilonGreedyPolicy(epsilon)
        bandit = GaussianBandit(n_list[i])
        agent = Agent(n_list[i], policy, num_trials)
        env = Environment(bandit, agent, num_trials, num_sessions)
        rewards[i, :, :], num_best[i, :, :] = env.run()

    # Compare average reward across values of epsilon
    color = iter(cm.rainbow(np.linspace(0, 1, len(n_list))))
    for i in range(len(n_list)):
        c = next(color)
        ave_reward = rewards[i, :, :].mean(axis=0)
        plt.plot(ave_reward, label="n:" + str(n_list[i]), c=c)
        plt.title("Average Reward")
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.legend(loc="upper left")
    plt.show()

    color2 = iter(cm.rainbow(np.linspace(0, 1, len(n_list))))
    for i in range(len(n_list)):
        c = next(color2)
        ave_percent_best = num_best[i, :, :].mean(axis=0)
        plt.plot(ave_percent_best, label="n:" + str(n_list[i]), c=c)
        plt.title("Average Percent Best Option")
        plt.xlabel('Trial')
        plt.ylabel('Percent Best Option')
        plt.legend(loc="upper left")
    plt.show()

    for i in range(len(n_list)):
        for j in range(num_sessions):
            cum_reward[j] = rewards[i, j, :].sum()
        ave_cum_reward[i, :] = [n_list[i], np.mean(cum_reward)]
    print(np.shape(cum_reward))
    print(np.shape(ave_cum_reward))
    print(ave_cum_reward)


def plot_ave_reward(rewards):
    # Average reward versus trials averaged over sessions
    ave_reward = rewards.mean(axis=0)
    plt.plot(ave_reward)
    plt.title("Average Reward")
    plt.xlabel('Trial')
    plt.ylabel('Reward')

    return ave_reward


def plot_percent_best_action(num_best):
    # Percent best action versus trials averaged over sessions for each time step
    ave_percent_best = num_best.mean(axis=0)
    plt.plot(ave_percent_best)
    plt.title("Average Percent Best Option")
    plt.xlabel('Trial')
    plt.ylabel('Percent Best Option')


def run_bandit(epsilon, n, num_trials, num_sessions):
    # Runs the bandit for a single epsilon, n
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

    random.seed(3)



    # RUN FOR A SINGLE EPSILON AND N
    # epsilon = 0.1
    # n = 10
    # num_trials = 500
    # num_sessions = 2000
    # run_bandit(epsilon, n, num_trials, num_sessions)

    # COMPARE EPSILONS FOR EPSILON GREEDY
    # epsilons = np.array((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    # epsilons = np.array((0, 0.2, 0.4, 0.6, 0.8, 1))
    # epsilons = np.array((0, 1))

    # COMPARE N FOR EPSILON GREEDY
    # n_list = np.array((2, 5, 10, 20))
    # compare_n(n_list)

    # COMPARE N AND EPSILON
    # compare_epsilons(2, epsilons)
    # compare_epsilons(5, epsilons)
    # compare_epsilons(10, epsilons)
    # compare_epsilons(20, epsilons)


