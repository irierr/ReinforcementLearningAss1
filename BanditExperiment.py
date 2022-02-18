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
def run_repetitions(method='egreedy', n_timesteps=1000, n_actions=10, epsilon=None, c=None):
    # initialize bandit environment
    env = BanditEnvironment(n_actions=n_actions)

    # greedy case
    if method == 'egreedy':
        pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
        rewards = []
        # iterate over time steps and save reward in every step
        for i in range(n_timesteps):
            a = pi.select_action(epsilon)  # select action
            r = env.act(a)  # sample reward
            rewards.append(r)
            pi.update(a, r)  # update policy

        return np.array(rewards)

    # OI case

    # ucb case
    if method == 'ucb':
        pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
        rewards = []
        # iterate over time steps and save reward in every step
        for i in range(n_timesteps):
            a = pi.select_action(c, i)  # select action
            r = env.act(a)  # sample reward
            rewards.append(r)
            pi.update(a, r)  # update policy
        return np.array(rewards)


def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    # TODO: Write all your experiment code here

    ### Only has 1 experiment per method yet, maybe we should code it more elegant to have every experiment inside a loop ###

    # Assignment 1: e-greedy
    r_means = np.zeros(n_timesteps)
    for i in range(n_repetitions):
        # print(run_repetitions())
        r_means += run_repetitions('egreedy', n_timesteps, n_actions, epsilon=0.1)
    r_means /= n_repetitions
    r_means = smooth(r_means, smoothing_window)
    Egreedy_plot = LearningCurvePlot(title="Egreedy")
    Egreedy_plot.add_curve(r_means)

    # Assignment 2: Optimistic init

    # Assignment 3: UCB
    r_means = np.zeros(n_timesteps)
    for i in range(n_repetitions):
        # print(run_repetitions())
        r_means += run_repetitions('ucb', n_timesteps, n_actions, c=0.1)
    r_means /= n_repetitions

    r_means = smooth(r_means, smoothing_window)
    Egreedy_plot = LearningCurvePlot(title="UCB")
    Egreedy_plot.add_curve(r_means)

    # Assignment 4: Comparison

    # pass


if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31

    experiment(n_actions=n_actions, n_timesteps=n_timesteps,
               n_repetitions=n_repetitions, smoothing_window=smoothing_window)
