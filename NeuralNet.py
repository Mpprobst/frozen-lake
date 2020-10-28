"""
NeuralNet.py
Author: Michael Probst
Purpose: Implements the Q-Learning model-free RL technique using a neural network to solve the frozen lake problem
Reference: https://www.youtube.com/watch?v=wc-FxNENg9U&feature=youtu.be
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""
import gym
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Generally, a NN is done by
1. Input to NN
2. Calculate output and feed forward (forward() method)
3. Backprop and update weights of the NN
"""

GAMMA = 0.99
LEARNING_RATE = 0.01

class Net(nn.Module):
    def __init__(self, inputDims, outputDims):
        super(Net, self).__init__()
        self.outputDims = outputDims
        self.inputDims = inputDims
        self.fc1 = nn.Linear(self.inputDims, self.inputDims)    #first layer
        self.fc2 = nn.Linear(self.inputDims, self.inputDims)    #second layer
        self.fc3 = nn.Linear(self.inputDims, self.outputDims)   #output layer
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
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.memorySize = 1000000
        self.memoryCounter = 0
        self.batchSize = 64
        self.net = Net(self.n_states, self.n_actions)
        self.optimizer = optim.SGD(self.net.parameters(), lr=LEARNING_RATE)
        self.successCount = 0
        self.qTable = np.zeros([self.n_states, env.action_space.n])

    def GetBestAction(self, state):
        oneHot = np.zeros(self.n_states, dtype=np.float32)
        oneHot[state] = 1.0
        s = T.tensor(oneHot).to(self.net.device).float()
        actions = self.net.Forward(s)
        self.recentQs = actions

        action = actions[T.argmax(actions).item()].item()
        options = []
        for i in range(len(actions)):
            if actions[i].item() == action:
                options.append(i)
        return random.choice(options)

    def EpsilonGreedy(self, env, state):
        epsilon = (0.5 / np.exp(0.01 * self.successCount))      # approximately .1 around successCount = 25,000 and e < 0.2 around 6000 This is because I am running 50,000 episodes and the model should be be able to have won half of the time by the end of training
        # explore
        if random.random() < epsilon:
            return env.action_space.sample()

        # exploit - get best action based on the state
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        return self.EpsilonGreedy(env, state)

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1

        for i in range(len(self.recentQs)):
            self.qTable[state][i] = self.recentQs[i]

        #backprop
        nextBestAction = self.GetBestAction(nextState)
        target = T.zeros(self.n_actions, device=self.net.device)
        target[action] = reward + GAMMA * self.qTable[nextState][nextBestAction]

        if nextState in self.terminalStates:
            target[action] = reward

        predict = T.zeros(self.n_actions, device=self.net.device)
        predict[action] = self.qTable[state][action]

        self.net.zero_grad()
        self.optimizer.zero_grad()
        # consider making the loss function an array size 4 with all 0's except for the index of taken action
        #loss = T.zeros((self.n_actions), device=self.net.device, requires_grad=True)
        #loss[action] = 0.5 * (target - predict)**2
        loss = F.smooth_l1_loss(predict, target)

        self.qTable[state][action] = loss.item()

        loss.requires_grad = True

        loss.backward()

        self.optimizer.step()
