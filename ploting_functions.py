import matplotlib.pyplot as plt
import numpy as np
from helper_functions_1_2_3 import *

# Question 1
def hist_heat_map(x_positions, y_positions, number_of_bins = 10):
    '''This is not njit'ed because it will not be a part of the parallisation'''
    plt.figure(figsize=(6,6))
    plt.hist2d(x_positions, y_positions, bins=number_of_bins, range=[[-1,1],[-1,1]], cmap="viridis")
    plt.colorbar(label="Number of samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Density map of sampled positions")
    plt.gca().set_aspect("equal")
    plt.show()

def plot_1_5(delta_arr):
    plt.figure(figsize=(7,5))

    plt.plot(
        delta_arr[:,0],
        delta_arr[:,1],
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Delta")
    plt.ylabel("Mean squared deviation of estimate of π")
    plt.title("Effect of Delta on π Estimate")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))

    plt.plot(
        delta_arr[:,0],
        delta_arr[:,2],
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Delta")
    plt.ylabel("Rejection rate")
    plt.title("Rejection Rate")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


# Question 2
def plot_q_2(std_dict, N, step):
    x_axis  = np.array(range(step,N,step))

    threshold = 0.001
    
    plt.figure()
    for key, value in std_dict.items():
        plt.plot(x_axis, value, label=key)
        if key == 'Importance Sampling std':
            ind = np.where(value < threshold)[0][0]
            plt.scatter(x_axis[ind], value[ind])
            plt.annotate(
                f"{x_axis[ind]} obs",
                (x_axis[ind], value[ind]),
                textcoords="offset points",
                xytext=(5,10)
            )
            print(f"We need at least {x_axis[ind]} observations for method 2 to reach std < {threshold}")
    plt.axhline(threshold, linestyle='--', label="Threshold (0.001)")

    plt.xlabel("Number of observations")
    plt.ylabel("Standard deviation")
    plt.title("Comparison of Standard Deviation Convergence")
    plt.grid(True)
    plt.legend()
    plt.show()

# Question 3
def plot_hist_3(Y_m):
    mu, std = norm.fit(Y_m)
    plt.hist(Y_m, bins=90, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, label='Y_m')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth=2, label='Normal distribution')
    title = "Y_m with mean: {:.2f} and standard deviation: {:.2f}".format(mu, std)
    plt.title(title)
    plt.xlabel('Bins') 
    plt.ylabel('Values') 
    plt.legend()
    plt.show()

def plot_grid_3_3(m_vals, n_vals):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for i, m in enumerate(m_vals):
        C_m_test = [(k - 0.5)/m for k in range(1, m+1)]
        for j, n in enumerate(n_vals):
            Y_m_test = exercise_3_2(m=m, n=n)
            Y_m_test.sort()
            y_grid = np.linspace(Y_m_test.min(), Y_m_test.max(), 500)
            mu = np.mean(Y_m_test)
            sigma = np.std(Y_m_test)
            F_clt = norm_cdf((y_grid-mu)/sigma)

            ax = axes[i, j]
            ax.plot(Y_m_test, C_m_test, label='Numerical CDF')
            ax.plot(y_grid, F_clt, '--', label='CLT CDF')
            Y_m_test.sort()
            ax.set_title(f"m={m}, n={n}")
            ax.set_xlabel("$Y_m$")
            ax.set_ylabel("$C_m$")

    plt.tight_layout()
    plt.show()


def plot_3_4(y_m_list, std_list, N_val):
    upper = y_m_list + 3 * std_list
    lower = y_m_list - 3 * std_list

    target_array = -2/np.pi**2*np.ones_like(y_m_list)

    fig, ax = plt.subplots(1, 3, figsize=(12,5))

    ax[0].plot(N_val, y_m_list, label="Estimate $Y_N$")
    ax[0].plot(N_val, target_array, linestyle='--', label="True value")
    ax[0].fill_between(N_val, lower, upper, alpha=0.3, label="±3 SE band")

    ax[0].set_xlabel("N")
    ax[0].set_ylabel("$Y_N$")
    ax[0].set_title("Full range")
    ax[0].legend()

    BEGIN = 25
    ax[1].plot(N_val[BEGIN:], y_m_list[BEGIN:], label="Estimate $Y_N$")
    ax[1].plot(N_val[BEGIN:], target_array[BEGIN:], linestyle='--', label="True value")
    ax[1].fill_between(N_val[BEGIN:], lower[BEGIN:], upper[BEGIN:], alpha=0.3, label="±3 SE band")

    ax[1].set_xlabel("N")
    ax[1].set_ylabel("$Y_N$")
    ax[1].set_title("Zoomed (large N)")
    ax[1].legend()

    BEGIN = 500
    ax[2].plot(N_val[BEGIN:], y_m_list[BEGIN:], label="Estimate $Y_N$")
    ax[2].plot(N_val[BEGIN:], target_array[BEGIN:], linestyle='--', label="True value")
    ax[2].fill_between(N_val[BEGIN:], lower[BEGIN:], upper[BEGIN:], alpha=0.3, label="±3 SE band")

    ax[2].set_xlabel("N")
    ax[2].set_ylabel("$Y_N$")
    ax[2].set_title("Zoomed even more ( even larger N)")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

def plot_3_5_2(true_european, mc_european_list, std_euro_list, N_euro_val):
    upper = mc_european_list + 3 * std_euro_list
    lower = mc_european_list - 3 * std_euro_list
    upper = upper.reshape(len(mc_european_list))
    lower = lower.reshape(len(mc_european_list))

    target_array = true_european*np.ones_like(mc_european_list)

    fig, ax = plt.subplots(1, 3, figsize=(12,5))

    ax[0].plot(N_euro_val, mc_european_list, label="Estimate $Y_N$")
    ax[0].plot(N_euro_val, target_array, linestyle='--', label="True value")
    ax[0].fill_between(N_euro_val, lower, upper, alpha=0.3, label="±3 SE band")

    ax[0].set_xlabel("N")
    ax[0].set_ylabel("$Y_N$")
    ax[0].set_title("Full range")
    ax[0].legend()

    BEGIN = 25
    ax[1].plot(N_euro_val[BEGIN:], mc_european_list[BEGIN:], label="Estimate $Y_N$")
    ax[1].plot(N_euro_val[BEGIN:], target_array[BEGIN:], linestyle='--', label="True value")
    ax[1].fill_between(N_euro_val[BEGIN:], lower[BEGIN:], upper[BEGIN:], alpha=0.3, label="±3 SE band")

    ax[1].set_xlabel("N")
    ax[1].set_ylabel("$Y_N$")
    ax[1].set_title("Zoomed (large N)")
    ax[1].legend()

    BEGIN = 500
    ax[2].plot(N_euro_val[BEGIN:], mc_european_list[BEGIN:], label="Estimate $Y_N$")
    ax[2].plot(N_euro_val[BEGIN:], target_array[BEGIN:], linestyle='--', label="True value")
    ax[2].fill_between(N_euro_val[BEGIN:], lower[BEGIN:], upper[BEGIN:], alpha=0.3, label="±3 SE band")

    ax[2].set_xlabel("N")
    ax[2].set_ylabel("$Y_N$")
    ax[2].set_title("Zoomed even more ( even larger N)")
    ax[2].legend()

    plt.tight_layout()
    plt.show()
