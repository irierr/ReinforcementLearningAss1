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
        # TO DO: Add own code
        pass
        
    def select_action(self, epsilon):
        # TO DO: Add own code
        a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        pass

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        # TO DO: Add own code
        pass
        
    def select_action(self):
        # TO DO: Add own code
        a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        pass

class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # TO DO: Add own code
        pass
    
    def select_action(self, c, t):
        # TO DO: Add own code
        a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        pass
    
def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
    
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(epsilon=0.5) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    
    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))
    
if __name__ == '__main__':
    test()
