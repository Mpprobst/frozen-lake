"""
DynaQ.py
Author: Michael Probst
Purpose: Implements the DynaQ model-based RL technique to solve the frozen lake problem
"""

class DynaQAgent:
    def __init__(self):
        self.policy = 0

    def SuggestMove(self, env):
        return env.action_space.sample()
