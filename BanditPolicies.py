#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment


class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.mean_estimates = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
        pass

    def select_action(self, epsilon):
        p = np.random.uniform()
        if p <= 1 - epsilon:  # with 1-ε we choose one of the action that maximizes mean_estimates(a)
            choices = np.argwhere(self.mean_estimates == np.max(self.mean_estimates)).flatten()
            a = np.random.choice(choices)
        else:  # otherwise we choose another action
            choices = np.argwhere(self.mean_estimates != np.max(self.mean_estimates)).flatten()
            if choices.size > 0:
                a = np.random.choice(choices)
            else:  # if this has no cases it means every action had the same Q value
                a = np.random.randint(self.n_actions)
        return a

    def update(self, a, r):
        # update rule for n(a) and mean_estimates(a)
        self.n[a] += 1
        self.mean_estimates[a] += (r - self.mean_estimates[a]) / self.n[a]
        return self


class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.mean_estimates = np.full(n_actions, initial_value)  # set all mean estimates to initial value
        pass

    def select_action(self):
        choices = np.argwhere(self.mean_estimates == np.max(self.mean_estimates)).flatten()
        a = np.random.choice(choices)
        return a

    def update(self, a, r):
        # update mean estimate with learning based update mean.
        self.mean_estimates[a] += self.learning_rate * (r - self.mean_estimates[a])
        pass


class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.mean_estimates = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
        pass

    def select_action(self, c, t):
        # set the function to optimize action, **important** keep eye on division by 0
        choices_0 = np.argwhere(self.n == 0).flatten()
        if choices_0.size > 0:
            choices = choices_0
        else:
            f = self.mean_estimates + c * np.sqrt(np.log(t + 1) / self.n)
            choices = np.argwhere(f == np.max(f)).flatten()
        a = np.random.choice(choices)
        return a

    def update(self, a, r):
        # update rule
        self.n[a] += 1
        self.mean_estimates[a] += (r - self.mean_estimates[a]) / self.n[a]


def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions)  # Initialize environment

    pi = EgreedyPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(epsilon=0.5)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a, r))

    pi = OIPolicy(n_actions=n_actions, initial_value=1.0)  # Initialize policy
    a = pi.select_action()  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a, r))

    pi = UCBPolicy(n_actions=n_actions)  # Initialize policy
    a = pi.select_action(c=1.0, t=1)  # select action
    r = env.act(a)  # sample reward
    pi.update(a, r)  # update policy
    print("Test UCB policy with action {}, received reward {}".format(a, r))


if __name__ == '__main__':
    test()
