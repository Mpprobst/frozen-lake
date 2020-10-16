"""
DynaQ.py
Author: Michael Probst
Purpose: Implements the DynaQ model-based RL technique to solve the frozen lake problem
"""


"""
QUESTIONS
Is my goal to train the model, then run some tests? - yes
Where do I get the mean reward?
    - train for some episodes, test. Train a little more, test. When testing, get the average reward.
Should I stop training when sufficiently well trained?
"""

import numpy as np
import random

LEARNING_RATE = 0.5
GAMMA = 0.1

"uses q-planning and q-learing"
class DynaQAgent:
    def __init__(self, env):
        "The transition model is initialized with every possible state and every"
        "possible move from that state with the state that results in that action"
        self.transitionModel = []      # [S, A] = S', R
        for s in range(env.observation_space.n):
            col = []
            for a in range(env.action_space.n):
                row = []
                for S in range(env.observation_space.n):
                    row.append(0)
                col.append(row)
            self.transitionModel.append(col)

        "The reward model contains the reward for every state"
        self.rewardModel = []           # [S, A] = R
        for s in range(env.observation_space.n):
            row = []
            for a in range(env.action_space.n):
                row.append(0)
            self.rewardModel.append(row)

        "Previously taken actions from a state"
        self.history = []               # [S, A] consider removing dupes by making a list
        self.successCount = 0

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
        epsilon = 1 / np.exp(0.1 * self.successCount)     # approximately 0 around n = 50
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
        self.history.append((state, action, nextState))
        # q update equation
        nextBestAction = self.GetBestAction(nextState)

        self.rewardModel[state][action] = self.rewardModel[state][action] + LEARNING_RATE * (reward + GAMMA * self.rewardModel[nextState][nextBestAction] - self.rewardModel[state][action])
        #print(f'{self.rewardModel[state][action]}+{LEARNING_RATE}*({reward}+{GAMMA}*{self.rewardModel[nextState][bestAction]}-{self.rewardModel[state][action]}={self.rewardModel[state][action]})')
        # update transition model with MLE
        successfulActionCount = 0
        actionTakenCount = 0
        for h in self.history:
            if h[0] == state and h[1] == action:
                actionTakenCount += 1
                if h[2] == nextState:
                    successfulActionCount += 1

        self.transitionModel[state][action][nextState] = successfulActionCount / actionTakenCount

        # q planning
        "for n times"
        for n in range(50):
            state, action, nextState = random.choice(self.history)
            bestAction = self.GetBestAction(nextState)
            sum_x = sum(self.transitionModel[state][action])
            randVal = random.uniform(0,sum_x)

            for s in range(len(self.transitionModel[state][action])):
                if self.transitionModel[state][action][s] < randVal:
                    nextState = s
                    break
                else:
                    randVal -= self.transitionModel[state][action][s]
            #print(nextState)
            #nextState = np.argmax(self.transitionModel[state][action])  # TODO: randomly select the nextState based on the probabilities.
            reward = self.transitionModel[state][action][nextState]

            # q update equation
            self.rewardModel[state][action] = self.rewardModel[state][action] + LEARNING_RATE * (reward + GAMMA * self.rewardModel[nextState][bestAction] - self.rewardModel[state][action])
