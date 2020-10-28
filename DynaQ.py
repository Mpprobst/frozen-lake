"""
DynaQ.py
Author: Michael Probst
Purpose: Implements the DynaQ model-based RL technique to solve the frozen lake problem
"""
"""
TEST 42.0:	 Avg Reward = 0.8 successCount = 5898 train time = 31.5400812625885
['LEFT ', '_UP_ ', '_UP_ ', '_UP_ ']
['LEFT ', 'LEFT ', 'RIGHT', 'LEFT ']
['_UP_ ', 'DOWN ', 'LEFT ', 'LEFT ']
['LEFT ', 'RIGHT', 'DOWN ', 'LEFT ']

"""
import numpy as np
import random

LEARNING_RATE = 0.001
GAMMA = 0.99
MAX_HISTORY_SIZE = 10000

"uses q-planning and q-learing"
class DynaQAgent:
    def __init__(self, env, terminalStates):
        "The transition model is initialized with every possible state and every"
        "possible move from that state with the state that results in that action"
        self.epsilon = 1
        self.terminalStates = terminalStates
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.transitionModel = np.zeros((n_states, n_actions, n_states), dtype=np.float32)     # [S, A, S'] = p

        "The q table contains the policy for each state"
        self.qTable = np.zeros((n_states, n_actions), dtype=np.float32)          # [S, A] = R

        self.rewardModel = np.zeros(n_states, dtype=np.float32)
        "Previously taken actions from a state"
        self.history = []              # (S, A, S')
        self.historyCounter = 0
        self.successCount = 0

    def GetBestAction(self, state):
        actions = self.qTable[state]
        bestAction = np.argmax(actions)
        bestReward = self.qTable[state][bestAction]
        "need to see if other actions have the same reward and pick one randomly if so"
        options = []
        for i in range(len(actions)):
            if actions[i] == bestReward:
                options.append(i)
        bestAction = random.choice(options)
        return bestAction

    def EpsilonGreedy(self, env, state):
        #epsilon = (1 / np.exp(0.001 * self.successCount)) + 0.01    # approximately 0 around successCount = 7,600 and e < 0.1 around 2,300
        # explore
        if random.random() < 0.1:#self.epsilon:
            return env.action_space.sample()

        # exploit - get best action based on the state
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        action = self.EpsilonGreedy(env, state)
        return action

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1
            self.epsilon -= 0.0005
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
        if self.historyCounter >= MAX_HISTORY_SIZE:
            self.historyCounter += 1
        if len(self.history) < MAX_HISTORY_SIZE:
            self.history.append((state, action, nextState))
        else:
            self.history[self.historyCounter] = (state, action, nextState)

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
        "for n times"
        for n in range(10):
            state, action, nextState = random.choice(limitedHistory)
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
            #print(f'{self.qTable[state][action]} = {predict}+{LEARNING_RATE}*({target}-{predict})')
