"""
DynaQ.py
Author: Michael Probst
Purpose: Implements the DynaQ model-based RL technique to solve the frozen lake problem
"""
import numpy as np
import random

LEARNING_RATE = 0.1
GAMMA = 0.9

TERMINAL_STATES = [5, 7, 11, 12, 15]

"uses q-planning and q-learing"
class DynaQAgent:
    def __init__(self, env):
        "The transition model is initialized with every possible state and every"
        "possible move from that state with the state that results in that action"
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
        self.history = []               # [S, A] consider removing dupes by making a list
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
        epsilon = (0.5 / np.exp(0.01 * self.successCount))      # approximately .1 around successCount = 25,000 and e < 0.2 around 6000 This is because I am running 50,000 episodes and the model should be be able to have won half of the time by the end of training
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
        #if (state,action,nextState) not in self.history:
        self.history.append((state, action, nextState))
        # q update equation
        nextBestAction = self.GetBestAction(nextState)
        target = reward + GAMMA * self.rewardModel[nextState][nextBestAction]
        if state in TERMINAL_STATES:
            target = reward
        self.rewardModel[state][action] = self.rewardModel[state][action] + LEARNING_RATE * (target - self.rewardModel[state][action])
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
        for n in range(5):
            state, action, nextState = random.choice(self.history)
            sum_x = sum(self.transitionModel[state][action])
            randVal = random.uniform(0,sum_x)

            # select nextState based on transition model probabilities
            for s in range(len(self.transitionModel[state][action])):
                if self.transitionModel[state][action][s] < randVal:
                    nextState = s
                    break
                else:
                    randVal -= self.transitionModel[state][action][s]

            reward = self.transitionModel[state][action][nextState]
            nextBestAction = self.GetBestAction(nextState)

            # q update equation
            "if in terminal then gamma*... = reward at next state"
            target = reward + GAMMA * self.rewardModel[nextState][nextBestAction]
            if state in TERMINAL_STATES:
                target = reward
            self.rewardModel[state][action] = self.rewardModel[state][action] + LEARNING_RATE * (target - self.rewardModel[state][action])
