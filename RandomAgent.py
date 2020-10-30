"""
RandomAgent.py
Author: Michael Probst
Purpose: Implements an agent that picks a random agent in the frozen lake environment
"""

import gym
import random
import numpy as np

class RandomAgent():
    def __init__(self, env, terminalStates=[]):
        self.qTable = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)     #only necessary to prevent errors while using verbose option
        self.successCount = 0

    def GetBestAction(self, state):
        return env.action_space.sample()

    def SuggestMove(self, env, state):
        return env.action_space.sample()

    # this is defined in other agents, so this must be defined to prevent errors
    def UpdateModels(self, state, nextState, action, reward):
        return
