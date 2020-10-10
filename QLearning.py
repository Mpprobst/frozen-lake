"""
QLearning.py
Author: Michael Probst
Purpose: Implements the q-learning model-free RL technique to solve the frozen lake problem
"""

class QLearningAgent:
    def __init__(self):
        self.policy = 0

    def SuggestMove(self, env):
        return env.action_space.sample()
