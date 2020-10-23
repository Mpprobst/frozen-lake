"""
DynaQ.py
Author: Michael Probst
Purpose: Implements the DynaQ model-based RL technique to solve the frozen lake problem
"""
import numpy as np
import random

LEARNING_RATE = 0.01
GAMMA = 0.6
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
        self.transitionModel = []     # [S, A, S'] = p
        for s in range(env.observation_space.n):
            col = []
            for a in range(env.action_space.n):
                row = []
                for S in range(env.observation_space.n):
                    row.append(0)
                col.append(row)
            self.transitionModel.append(col)


        "The reward model contains the reward for every state"
        self.rewardModel = []          # [S, A] = R
        for s in range(env.observation_space.n):
            row = []
            for a in range(env.action_space.n):
                row.append(0)
            self.rewardModel.append(row)

        "Previously taken actions from a state"
        self.history = []              # [S, A] consider removing dupes by making a list
        self.historyCounter = 0
        self.successCount = 0

    def GetBestAction(self, state):
        actions = self.rewardModel[state]
        bestAction = np.argmax(actions)
        bestReward = self.rewardModel[state][bestAction]
        "need to see if other actions have the same reward and pick one randomly if so"
        options = []
        for i in range(len(actions)):
            if actions[i] == bestReward:
                options.append(i)
        bestAction = random.choice(options)
        return bestAction

    def EpsilonGreedy(self, env, state):
        #epsilon = (1 / np.exp(0.01 * self.successCount)) + 0.01    # approximately 0 around successCount = 7,600 and e < 0.1 around 2,300
        # explore
        if random.random() < self.epsilon:
            return env.action_space.sample()

        # exploit - get best action based on the state
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        action = self.EpsilonGreedy(env, state)
        return action

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1
            self.epsilon -= 0.0001
            if self.epsilon < 0.01:
                self.epsilon = 0.01

        # q update equation
        nextBestAction = self.GetBestAction(nextState)
        target = reward + GAMMA * self.rewardModel[nextState][nextBestAction]
        if nextState in self.terminalStates:
            target = reward
        self.rewardModel[state][action] = self.rewardModel[state][action] + LEARNING_RATE * (target - self.rewardModel[state][action])
        #print(f'{self.rewardModel[state][action]}+{LEARNING_RATE}*({reward}+{GAMMA}*{self.rewardModel[nextState][bestAction]}-{self.rewardModel[state][action]}={self.rewardModel[state][action]})')

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
        #print("%.3f"%self.transitionModel[state][action][nextState])

        limitedHistory = list(dict.fromkeys(self.history))
        #print(f'{len(limitedHistory)}, {len(self.history)} ')
        # q planning
        "for n times"
        for n in range(5):
            state, action, nextState = random.choice(limitedHistory)
            sum_x = sum(self.transitionModel[state][action])
            randVal = random.uniform(0,sum_x)

            # select nextState based on transition model probabilities
            for s in range(len(self.transitionModel[state][action])):
                if self.transitionModel[state][action][s] < randVal:
                    nextState = s
                    break
                else:
                    randVal -= self.transitionModel[state][action][s]

            prob = self.transitionModel[state][action][nextState]
            nextBestAction = self.GetBestAction(nextState)

            # q update equation
            "if in terminal then gamma*... = reward at next state"
            target = prob + GAMMA * self.rewardModel[nextState][nextBestAction]
            if nextState in self.terminalStates:
                target = reward
            predict = self.rewardModel[state][action]
            self.rewardModel[state][action] = self.rewardModel[state][action] + LEARNING_RATE * (target - self.rewardModel[state][action])
            #print(f'{self.rewardModel[state][action]} = {predict}+{LEARNING_RATE}*({target}-{predict})')
