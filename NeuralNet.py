"""
NeuralNet.py
Author: Michael Probst
Purpose: Implements the Q-Learning model-free RL technique using a neural network to solve the frozen lake problem
"""

class NeuralNetAgent:
    def __init__(self):
        self.policy = 0

    def SuggestMove(self, env):
        return env.action_space.sample()
