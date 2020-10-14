"""
DynaQ.py
Author: Michael Probst
Purpose: Implements the DynaQ model-based RL technique to solve the frozen lake problem
"""


"""
QUESTIONS
Is my goal to train the model, then run some tests?
Where do I get the mean reward
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
        actions = np.amax(self.transitionModel[state], axis=1)
        bestAction = np.argmax(actions)
        bestReward = self.transitionModel[state][bestAction][0]

        "need to see if other actions have the same reward"
        if np.array(actions==bestReward).sum() > 1:
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

        self.transitionModel[state][action][nextState] = reward

        # q planning
        "for n times"
        for n in range(100):
            state, action, nextState = random.choice(self.history)
            bestAction = self.GetBestAction(nextState)
            "randomAction = random action taken previously from randomState"
            reward = self.transitionModel[state][action][nextState]
            # q update equation
            self.rewardModel[state][action] = self.rewardModel[state][action] + LEARNING_RATE * (reward + GAMMA * self.rewardModel[nextState][bestAction] - self.rewardModel[state][action])
