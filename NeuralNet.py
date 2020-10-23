"""
NeuralNet.py
Author: Michael Probst
Purpose: Implements the Q-Learning model-free RL technique using a neural network to solve the frozen lake problem
Reference: https://www.youtube.com/watch?v=wc-FxNENg9U&feature=youtu.be
"""
import gym
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.9
LEARNING_RATE = 0.01

class DeepQNetwork(nn.Module):
    def __init__(self, inputDims, fc1Dims, fc2Dims, env):
        super(DeepQNetwork, self).__init__()
        self.n_actions = env.action_space.n
        self.inputDims = inputDims
        self.fc1Dims = fc1Dims
        self.fc2Dims = fc2Dims
        self.fc1 = nn.Linear(*self.inputDims, self.fc1Dims) #first layer
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)    #second layer
        self.fc3 = nn.Linear(self.fc2Dims, self.n_actions)  #output layer
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()    #mean squared error
        self.device = T.device('cpu')
        self.to(self.device)

    "Implements a feed forward network"
    def Forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class NeuralNetAgent:
    def __init__(self, env, terminalStates):
        self.terminalStates = terminalStates
        self.memorySize = 1000000
        self.memoryCounter = 0
        self.batchSize = 64
        inputDims = [4]
        self.qEval = DeepQNetwork(inputDims, 4, 4, env)
        self.stateMemory = np.zeros((self.memorySize, *inputDims), dtype=np.float32)
        self.nextStateMemory = np.zeros((self.memorySize, *inputDims), dtype=np.float32)
        self.actionMemory = np.zeros(self.memorySize, dtype=np.int32)
        self.rewardMemory = np.zeros(self.memorySize, dtype=np.float32)
        self.terminalMemory = np.zeros(self.memorySize, dtype=bool)
        self.successCount = 0

    def StoreTransiton(self, state, nextState, action, reward):
        index = self.memoryCounter % self.memorySize
        self.stateMemory[index] = state
        self.nextStateMemory[index] = nextState
        self.actionMemory[index] = action
        self.rewardMemory[index] = reward
        self.terminalMemory[index] = True if state in self.terminalStates else False
        self.memoryCounter += 1

    def GetBestAction(self, state):
        s = T.tensor([state,state,state,state]).to(self.qEval.device).float()
        actions = self.qEval.Forward(s)
        action = T.argmax(actions).item()
        return action

    def EpsilonGreedy(self, env, state):
        epsilon = (0.5 / np.exp(0.01 * self.successCount))      # approximately .1 around successCount = 25,000 and e < 0.2 around 6000 This is because I am running 50,000 episodes and the model should be be able to have won half of the time by the end of training
        # explore
        if random.random() < epsilon:
            return env.action_space.sample()

        # exploit - get best action based on the state
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        return self.EpsilonGreedy(env, state)

    def Learn(self):
        if self.memoryCounter < self.batchSize:
            return

        self.qEval.optimizer.zero_grad()

        maxMem = min(self.memoryCounter, self.memorySize)
        batch = np.random.choice(maxMem, self.batchSize, replace=False)
        batchIndex = np.arange(self.batchSize, dtype=np.int32)

        stateBatch = T.tensor(self.stateMemory[batch]).to(self.qEval.device)
        nextStateBatch = T.tensor(self.nextStateMemory[batch]).to(self.qEval.device)
        rewardBatch = T.tensor(self.rewardMemory[batch]).to(self.qEval.device)
        terminalBatch = T.tensor(self.terminalMemory[batch]).to(self.qEval.device)

        actionBatch = self.actionMemory[batch]
        #print(f'bi = {batchIndex} a = {actionBatch}')

        qEval = self.qEval.Forward(stateBatch)[batchIndex, actionBatch]
        qNext = self.qEval.Forward(nextStateBatch)
        qNext[terminalBatch] = 0.0
        qTarget = rewardBatch + GAMMA * T.max(qNext, dim=1)[0]
        loss = self.qEval.loss(qTarget, qEval).to(self.qEval.device)
        loss.backward()
        self.qEval.optimizer.step()

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1
        self.StoreTransiton(state, nextState, action, reward)
        self.Learn()
