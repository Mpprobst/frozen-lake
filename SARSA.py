"""
SARSA.py
Author: Michael Probst
Purpose: Implements the SARSA model-free RL technique to solve the frozen lake problem
"""

class SARSAAgent:
    def __init__(self):
        self.policy = 0

    def SuggestMove(self, env):
        return env.action_space.sample()
