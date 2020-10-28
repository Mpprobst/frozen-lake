"""
QLearning.py
Author: Michael Probst
Purpose: Implements the q-learning model-free RL technique to solve the frozen lake problem
"""
import numpy as np
import random

LEARNING_RATE = 0.5
GAMMA = 0.9

"""
['LEFT', 'UP', 'DOWN', 'UP']
['LEFT', 'LEFT', 'LEFT', 'LEFT']
['UP', 'DOWN', 'LEFT', 'LEFT']
['LEFT', 'RIGHT', 'UP', 'LEFT']
"""

class QLearningAgent:
    def __init__(self, env, terminalStates=[]):
        "The reward model contains the reward for every state"
        self.qTable = []           # [S, A] = R
        for s in range(env.observation_space.n):
            row = []
            for a in range(env.action_space.n):
                row.append(0)
            self.qTable.append(row)

        self.successCount = 0

    def GetBestAction(self, state):
        actions = self.qTable[state]
        bestAction = np.argmax(actions)
        bestReward = self.qTable[state][bestAction]
        "need to see if other actions have the same reward"
        options = []
        for i in range(len(actions)):
            if actions[i] == bestReward:
                options.append(i)
        bestAction = random.choice(options)
        return bestAction

    def EpsilonGreedy(self, env, state):
        epsilon = (0.5 / np.exp(0.01 * self.successCount))      # approximately 0 around successCount = 600 and e < 0.1 around 50
        # explore
        if random.random() < epsilon:
            return env.action_space.sample()

        # exploit - get best action based on the state
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
