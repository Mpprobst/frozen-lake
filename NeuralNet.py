"""
NeuralNet.py
Author: Michael Probst
Purpose: Implements the Q-Learning model-free RL technique using a neural network to solve the frozen lake problem
"""
import gym
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.98
LEARNING_RATE = 0.3

class Net(nn.Module):
    def __init__(self, inputDims, outputDims):
        super(Net, self).__init__()
        self.outputDims = outputDims
        self.inputDims = inputDims
        self.fc1 = nn.Linear(self.inputDims, 32)    #first layer
        self.fc2 = nn.Linear(32, 16)                #second layer
        self.fc3 = nn.Linear(16, self.outputDims)   #output layer
        self.device = T.device('cpu')
        self.to(self.device)

    #Implements a feed forward network. state is a one hot vector indicating current state
    def Forward(self, state):
        x = F.logsigmoid(self.fc1(state))
        x = F.logsigmoid(self.fc2(x))
        actions = self.fc3(x)
        return actions

class NeuralNetAgent:
    def __init__(self, env, terminalStates):
        self.epsilon = 1
        self.terminalStates = terminalStates
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.net = Net(self.n_states, self.n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.qTable = np.zeros([self.n_states, env.action_space.n])     # for the NN, this is for debugging only
        self.successCount = 0

    def GetStateTensor(self, state):
        oneHot = np.zeros(self.n_states, dtype=np.float32)
        oneHot[state] = 1.0
        return T.tensor(oneHot).to(self.net.device).float()

    def GetBestAction(self, state):
        stateTensor = self.GetStateTensor(state)
        actions = self.net.Forward(stateTensor)
        self.recentQs = actions

        action = actions[T.argmax(actions).item()].item()
        options = []
        for i in range(len(actions)):
            if actions[i].item() == action:
                options.append(i)
        return random.choice(options)

    def EpsilonGreedy(self, env, state):
        # explore
        if random.random() < self.epsilon:
            return env.action_space.sample()

        # exploit
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        return self.EpsilonGreedy(env, state)

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1
            self.epsilon -= 0.002
            if self.epsilon < 0.01:
                self.epsilon = 0.01
        for n in range(5):
            self.Train(state, nextState, action, reward)

    def Train(self, state, nextState, action, reward):
        # updates a qTable with previously calculated Q values from the NN. Used for debug only.
        for i in range(len(self.recentQs)):
            self.qTable[state][i] = self.recentQs[i]

        #backprop
        predict = T.zeros(self.n_actions, device=self.net.device)
        stateTensor = self.GetStateTensor(state)
        predict[action] = self.net.Forward(stateTensor)[action]

        target = T.zeros(self.n_actions, device=self.net.device)
        nextBestAction = self.GetBestAction(nextState)
        nextStateTensor = self.GetStateTensor(nextState)
        target[nextBestAction] = reward + GAMMA * self.net.Forward(nextStateTensor)[nextBestAction]

        if nextState in self.terminalStates:
            target[nextBestAction] = reward

        loss = F.mse_loss(predict, target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
