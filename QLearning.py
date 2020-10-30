"""
QLearning.py
Author: Michael Probst
Purpose: Implements the q-learning model-free RL technique to solve the frozen lake problem
"""
import numpy as np
import random

LEARNING_RATE = 0.15
GAMMA = 0.9

"""
['LEFT', 'UP', 'DOWN', 'UP']
['LEFT', 'LEFT', 'LEFT', 'LEFT']
['UP', 'DOWN', 'LEFT', 'LEFT']
['LEFT', 'RIGHT', 'UP', 'LEFT']
"""

class QLearningAgent:
    def __init__(self, env, terminalStates=[]):
        self.terminalStates = terminalStates
        self.qTable = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)           # [S, A] = R
        self.successCount = 0

    def GetBestAction(self, state):
        actions = self.qTable[state]
        bestAction = np.argmax(actions)
        bestReward = self.qTable[state][bestAction]

        #break ties randomly if ties exist
        options = []
        for i in range(len(actions)):
            if actions[i] == bestReward:
                options.append(i)
        bestAction = random.choice(options)
        return bestAction

    def EpsilonGreedy(self, env, state):
        epsilon = (0.5 / np.exp(0.01 * self.successCount))      # approximately 0 around successCount = 600
        # explore
        if random.random() < epsilon:
            return env.action_space.sample()

        # exploit
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        action = self.EpsilonGreedy(env, state)
        return action

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1

        nextBestAction = self.GetBestAction(nextState)
        # q update equation
        self.qTable[state][action] = self.qTable[state][action] + LEARNING_RATE * (reward + GAMMA * self.qTable[nextState][nextBestAction] - self.qTable[state][action])
