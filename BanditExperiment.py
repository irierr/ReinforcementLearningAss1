#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
from tkinter import E
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth


# keep an eye on this function, should be general to all methods to run experiments
def run_repetitions(method='egreedy', n_timesteps=1000, n_actions=10, policy_parameter=None):
    # initialize bandit environment
    env = BanditEnvironment(n_actions=n_actions)
    rewards = []

    # greedy case
    if method == 'egreedy':
        pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
        # iterate over time steps and save reward in every step
        for i in range(n_timesteps):
            a = pi.select_action(policy_parameter)  # select action
            r = env.act(a)  # sample reward
            rewards.append(r)
            pi.update(a, r)  # update policy

    # OI case
    elif method == 'oi':
        pi = OIPolicy(n_actions=n_actions, initial_value=policy_parameter)
        for i in range(n_timesteps):
            a = pi.select_action()  # select action
            r = env.act(a)  # sample reward
            rewards.append(r)
            pi.update(a, r)  # update policy

    # ucb case
    elif method == 'ucb':
        pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
        # iterate over time steps and save reward in every step
        for i in range(n_timesteps):
            a = pi.select_action(policy_parameter, i)  # select action
            r = env.act(a)  # sample reward
            rewards.append(r)
            pi.update(a, r)  # update policy

    return np.array(rewards)


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    # Assignment 1: e-greedy
    # TODO: Give proper title
    epsilon_values = [0.01, 0.05, 0.1, 0.25]
    e_greedy_plot = LearningCurvePlot(title="comparison of ε values in ε-greedy")
    e_greedy_rewards = np.zeros(shape=(len(epsilon_values), n_timesteps))
    for e_index, e in enumerate(epsilon_values):
        r_means = np.zeros(n_timesteps)
        for i in range(n_repetitions):
            r_means += run_repetitions('egreedy', n_timesteps, n_actions, e)
        r_means /= n_repetitions
        r_means = smooth(r_means, smoothing_window)
        e_greedy_plot.add_curve(r_means, str(e))
        e_greedy_rewards[e_index] = r_means
    e_greedy_plot.save("e_greedy_plot.png", legend_title="ε Value")

    # Assignment 2: Optimistic init
    # TODO: Give proper title
    initial_values = [0.1, 0.5, 1.0, 2.0]
    oi_plot = LearningCurvePlot(title="OI")
    oi_rewards = np.zeros(shape=(len(initial_values), n_timesteps))
    for val_index, val in enumerate(initial_values):
        r_means = np.zeros(n_timesteps)
        for i in range(n_repetitions):
            r_means += run_repetitions('oi', n_timesteps, n_actions, val)
        r_means /= n_repetitions
        r_means = smooth(r_means, smoothing_window)
        oi_rewards[val_index] = r_means
        oi_plot.add_curve(r_means, str(val))
    oi_plot.save("oi_plot.png", legend_title="Initial Mean Estimate")

    # Assignment 3: UCB
    # TODO: Give proper title
    c_values = [.01, .05, .1, .25, .5, 1]
    ucb_plot = LearningCurvePlot(title="UCB")
    ucb_rewards = np.zeros(shape=(len(c_values), n_timesteps))
    for index_c, c in enumerate(c_values):
        r_means = np.zeros(n_timesteps)
        for i in range(n_repetitions):
            r_means += run_repetitions('ucb', n_timesteps, n_actions, c)
        r_means /= n_repetitions
        r_means = smooth(r_means, smoothing_window)
        ucb_rewards[index_c] = r_means
        ucb_plot.add_curve(r_means, str(c))
    ucb_plot.save('ucb_plot.png', legend_title="Exploration Constant")

    # Assignment 4: Comparison
    # TODO: Give proper title
    comparison_plot = ComparisonPlot(title="Comparison", timesteps=n_timesteps)
    e_greedy_mean_rewards = np.mean(e_greedy_rewards, axis=1)
    oi_mean_rewards = np.mean(oi_rewards, axis=1)
    ucb_mean_rewards = np.mean(ucb_rewards, axis=1)
    comparison_plot.add_curve(x=epsilon_values, y=e_greedy_mean_rewards, label="ε-Greedy")
    comparison_plot.add_curve(x=initial_values, y=oi_mean_rewards, label="Optimistic Initialization")
    comparison_plot.add_curve(x=c_values, y=ucb_mean_rewards, label='UCB')
    comparison_plot.save(name="Comparison", legend_title="Policies")

    # TODO: Give proper title
    optimal_plot = LearningCurvePlot()
    opt_epsilon_index = np.argmax(e_greedy_mean_rewards)
    opt_initial_val_index = np.argmax(oi_mean_rewards)
    opt_c_index = np.argmax(ucb_mean_rewards)
    optimal_plot.add_curve(e_greedy_rewards[opt_epsilon_index], label='ε-Greedy')
    optimal_plot.add_curve(oi_rewards[opt_initial_val_index], label='Optimistic Initialization')
    optimal_plot.add_curve(ucb_rewards[opt_c_index], label='UCB')
    optimal_plot.save("optimal_plots.png", legend_title="Policies")
    # pass


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 50
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
