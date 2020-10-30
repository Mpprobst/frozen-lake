"""
DynaQ.py
Author: Michael Probst
Purpose: Implements the DynaQ model-based RL technique to solve the frozen lake problem
"""
import numpy as np
import random

LEARNING_RATE = 0.001
GAMMA = 0.99
MAX_HISTORY_SIZE = 5000

class DynaQAgent:
    def __init__(self, env, terminalStates):
        self.epsilon = 1
        self.terminalStates = terminalStates
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.transitionModel = np.zeros((n_states, n_actions, n_states), dtype=np.float32)     # [S, A, S'] = p
        self.qTable = np.zeros((n_states, n_actions), dtype=np.float32)                         # [S, A] = R
        self.rewardModel = np.zeros(n_states, dtype=np.float32)
        self.history = []              # (S, A, S')
        self.historyCounter = 0
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
        # explore
        if random.random() < self.epsilon:
            return env.action_space.sample()

        # exploit
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        action = self.EpsilonGreedy(env, state)
        return action

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1
            self.epsilon -= 0.001
            if self.epsilon < 0.01:
                self.epsilon = 0.01

        self.rewardModel[nextState] = reward

        # q update equation
        nextBestAction = self.GetBestAction(nextState)
        target = reward + GAMMA * self.qTable[nextState][nextBestAction]
        if nextState in self.terminalStates:
            target = reward
        self.qTable[state][action] = self.qTable[state][action] + LEARNING_RATE * (target - self.qTable[state][action])

        # add to history array
        if len(self.history) < MAX_HISTORY_SIZE:
            self.history.append((state, action, nextState))
        else:
            self.history[self.historyCounter] = (state, action, nextState)

        self.historyCounter += 1
        if self.historyCounter >= MAX_HISTORY_SIZE:
            self.historyCounter = 0

        # update transition model with MLE
        successfulActionCount = 0
        actionTakenCount = 0
        for h in self.history:
            if h[0] == state and h[1] == action:
                actionTakenCount += 1
                if h[2] == nextState:
                    successfulActionCount += 1
        self.transitionModel[state][action][nextState] = successfulActionCount / actionTakenCount

        limitedHistory = list(dict.fromkeys(self.history))

        # q planning
        for n in range(10):
            state, action, nextState = random.choice(limitedHistory)
            # we dont make an action if we are in a terminal state so skip this iteration
            if state in self.terminalStates:
                continue

            sum_x = sum(self.transitionModel[state][action])
            randVal = random.uniform(0,sum_x)

            # select nextState based on transition model probabilities
            for s in range(len(self.transitionModel[state][action])):
                if self.transitionModel[state][action][s] < randVal:
                    nextState = s
                    break
                else:
                    randVal -= self.transitionModel[state][action][s]

            nextBestAction = self.GetBestAction(nextState)

            # q update equation
            target = reward + GAMMA * self.qTable[nextState][nextBestAction]
            if nextState in self.terminalStates:
                target = self.rewardModel[nextState]
            self.qTable[state][action] = self.qTable[state][action] + LEARNING_RATE * (target - self.qTable[state][action])
