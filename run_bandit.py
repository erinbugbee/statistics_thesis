import numpy as np

n = 10
num_sessions = 100
num_trials = 500
val_std = 1
obs_std = 3
epsilon = 0.1
opt_LR = True
invT_vec = np.linspace(0, 10, 10)
fixed_LR_vec = np.linspace(0, 10, 10)
num_changes = 5
optLR_reset = True
qvals_reset = True
selection = "epsilon_greedy"


def action_selection(selection):
    if selection == "epsilon_greedy":
        return 0
    elif selection == "softmax":
        return 1


def bandit(val_vec, transition_times, fixed_LR = None):
    arms = np.arange(n)
    current_vals = val_vec[:, 0]
    for session in range(num_sessions):
        reward = np.zeros((1, num_trials))
        average_reward = np.zeros((1, num_trials))
        k = np.zeros((n, 1))
        Q = np.zeros((n, num_trials))
        Q_stored = np.zeros((n, len(transition_times)))
        for trial in range(num_trials-1):
            if trial in transition_times:
                index = np.where(transition_times == trial)[0]
                current_vals = val_vec[:, (index + 1)]
                if optLR_reset:
                    k = np.zeros((n, 1))
                if qvals_reset:
                    Q_col = np.reshape(Q[:, trial], (10, 1))
                    Q_stored[:, index] = Q_col
                    Q[:, trial] = np.zeros((n))
            chosen_action = action_selection("epsilon_greedy")
            reward[0, trial] = np.random.normal(current_vals[chosen_action], obs_std)
            k[chosen_action, 0] = k[chosen_action, 0] + 1

            for a in range(n):
                if a == chosen_action:
                    if opt_LR:
                        LR = 1/k[a, 0]
                    else:
                        LR = fixed_LR_vec
                    Q[a, trial + 1] = Q[a, trial] + LR*(reward[0, trial] - Q[a, trial])
                else:
                    Q[a, trial + 1] = Q[a, trial]



def main():
    if opt_LR:
        performance_mat = np.zeros((len(invT_vec), num_trials))
    else:
        performance_mat = np.zeros((len(invT_vec), len(fixed_LR_vec), num_trials))
    transition_times = []
    curr_transition = 0
    val_vec = []
    if num_changes != 0:
        transition_diff = round(num_trials / (num_changes + 1))
        while (curr_transition + transition_diff) < num_trials:
            transition_times = np.append(transition_times, curr_transition + transition_diff)
            curr_transition = curr_transition + transition_diff
    val_vec = np.random.normal(0, val_std, (n, len(transition_times) + 1))
    for i in range(len(invT_vec)):
        if opt_LR:
            performance = bandit(val_vec, transition_times)
            performance_mat[i, :] = performance
        else:
            for j in range(len(fixed_LR_vec)):
                performance = bandit(val_vec, transition_times, fixed_LR_vec[j])
                performance_mat[i, j, :] = performance
    # Do some averaging here
    return


if __name__ == '__main__':
    main()
