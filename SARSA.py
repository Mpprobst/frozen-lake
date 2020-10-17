"""
SARSA.py
Author: Michael Probst
Purpose: Implements the SARSA model-free RL technique to solve the frozen lake problem
"""
import numpy as np
import random

LEARNING_RATE = 0.9
GAMMA = 0.9

class SARSAAgent:
    def __init__(self, env):
        "The reward model contains the reward for every state"
        self.rewardModel = []           # [S, A] = R
        for s in range(env.observation_space.n):
            row = []
            for a in range(env.action_space.n):
                row.append(0)
            self.rewardModel.append(row)

        self.successCount = 0
        self.previousObservation = None

    def GetBestAction(self, state):
        actions = self.rewardModel[state]
        bestAction = np.argmax(actions)
        bestReward = self.rewardModel[state][bestAction]
        "need to see if other actions have the same reward"
        options = []
        for i in range(len(actions)):
            if actions[i] == bestReward:
                options.append(i)
        bestAction = random.choice(options)
        return bestAction

    def EpsilonGreedy(self, env, state):
        epsilon = (1 / np.exp(0.1 * self.successCount))      # approximately 0 around successCount = 600 and e < 0.1 around 50
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
        if self.previousObservation != None:
            # q update equation
            s, a = self.previousObservation
            self.rewardModel[s][a] = self.rewardModel[s][a] + LEARNING_RATE * (reward + GAMMA * self.rewardModel[state][action] - self.rewardModel[s][a])
        self.previousObservation = (state, action)
