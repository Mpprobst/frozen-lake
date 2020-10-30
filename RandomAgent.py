"""
RandomAgent.py
Author: Michael Probst
Purpose: Implements an agent that picks a random agent in the frozen lake environment
"""

import gym
import random

class RandomAgent():
    def __init__(self):
        self.successCount = 0

    def SuggestMove(self, env, terminalStates=[]):
        return env.action_space.sample()
