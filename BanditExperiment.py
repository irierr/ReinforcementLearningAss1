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


def run_each_parameter(method, parameter_values, n_actions, n_timesteps, n_repetitions, smoothing_window) -> np.array:
    rewards = np.zeros(shape=(len(parameter_values), n_timesteps))
    for i, value in enumerate(parameter_values):
        r_means = np.zeros(n_timesteps)
        for j in range(n_repetitions):
            r_means += run_repetitions(method, n_timesteps, n_actions, value)
        r_means /= n_repetitions
        r_means = smooth(r_means, smoothing_window)
        rewards[i] = r_means
    return rewards


def plot_exp(title: str, legend_title: str, rewards, legend_values):
    plot = LearningCurvePlot("")
    for i, value in enumerate(legend_values):
        plot.add_curve(rewards[i], str(value))
    plot.save(title.replace(" ", "_"), legend_title)


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    # TODO: Give proper titles to graphs
    # Assignment 1: e-greedy
    epsilon_values = [0.01, 0.05, 0.1, 0.25]
    e_greedy_rewards = run_each_parameter("egreedy", epsilon_values, n_actions,
                                          n_timesteps, n_repetitions, smoothing_window)
    plot_exp(title="egreedy", legend_title="ε Value", rewards=e_greedy_rewards, legend_values=epsilon_values)

    # Assignment 2: Optimistic init
    initial_values = [0.1, 0.5, 1.0, 2.0]
    oi_rewards = run_each_parameter("oi", initial_values, n_actions, n_timesteps, n_repetitions, smoothing_window)
    plot_exp(title="oi", legend_title="Initial Mean Estimate", rewards=oi_rewards, legend_values=initial_values)

    # Assignment 3: UCB
    c_values = [.01, .05, .1, .25, .5, 1]
    ucb_rewards = run_each_parameter("ucb", c_values, n_actions, n_timesteps, n_repetitions, smoothing_window)
    plot_exp(title="ucb", legend_title="Exploration Constant", rewards=ucb_rewards, legend_values=c_values)

    # Assignment 4: Comparison
    comparison_plot = ComparisonPlot(title="")
    e_greedy_mean_rewards = np.mean(e_greedy_rewards, axis=1)
    oi_mean_rewards = np.mean(oi_rewards, axis=1)
    ucb_mean_rewards = np.mean(ucb_rewards, axis=1)
    comparison_plot.add_curve(x=epsilon_values, y=e_greedy_mean_rewards, label="ε-Greedy")
    comparison_plot.add_curve(x=initial_values, y=oi_mean_rewards, label="Optimistic Initialization")
    comparison_plot.add_curve(x=c_values, y=ucb_mean_rewards, label='UCB')
    comparison_plot.save(name="comparison", legend_title="Policies")

    optimal_plot = LearningCurvePlot(title="Average reward compared for 3 bandit policies with optimized parameters")
    opt_epsilon_index = np.argmax(e_greedy_mean_rewards)
    opt_initial_val_index = np.argmax(oi_mean_rewards)
    opt_c_index = np.argmax(ucb_mean_rewards)
    optimal_plot.add_curve(e_greedy_rewards[opt_epsilon_index], label='ε-Greedy')
    optimal_plot.add_curve(oi_rewards[opt_initial_val_index], label='Optimistic Initialization')
    optimal_plot.add_curve(ucb_rewards[opt_c_index], label='UCB')
    optimal_plot.save("optimal_plots", legend_title="Policies")


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
