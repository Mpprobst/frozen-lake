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

"""
TEST 99.0:	 Avg Reward = 0.0 successCount = 207 train time = 235.1845030784607
[0.0031, 'LEFT ', 0.003, 'LEFT ', 0.1481, 'DOWN ', 0.0, 'RIGHT']
[0.0, 'RIGHT', -0.0, '_UP_ ', 0.0, 'LEFT ', 0.1632, 'LEFT ']
[0.2813, 'DOWN ', 0.0, 'RIGHT', 0.1835, 'DOWN ', 0.0019, 'LEFT ']
[0.9533, 'DOWN ', 0.0145, 'LEFT ', 0.0376, 'LEFT ', 0.0, '_UP_ ']
[0.0002, '_UP_ ', 0.0, 'RIGHT', 0.1242, 'DOWN ', 0.0, 'LEFT ']
[0.5349, 'DOWN ', 0.5358, 'DOWN ', 0.0, 'DOWN ', 0.0, 'RIGHT']
[0.1735, 'LEFT ', 0.0001, 'DOWN ', -0.0, 'RIGHT', 0.0, '_UP_ ']
[0.0, 'RIGHT', 0.0, 'LEFT ', 0.0, 'DOWN ', -0.0, '_UP_ ']
[0.0002, 'DOWN ', 0.0002, 'DOWN ', 0.0002, 'DOWN ', 0.0, 'LEFT ']
[0.3968, 'LEFT ', 0.0183, 'DOWN ', 0.6354, '_UP_ ', 0.0, '_UP_ ']
[0.0, 'RIGHT', 0.0, 'LEFT ', 0.0, 'LEFT ', 0.0, 'LEFT ']
[0.352, 'LEFT ', 0.0001, 'RIGHT', 0.0, 'LEFT ', 0.1894, 'RIGHT']
[0.0, 'RIGHT', 0.0, 'LEFT ', -0.0, 'RIGHT', 0.0012, 'DOWN ']
[0.0, 'LEFT ', 0.0, 'RIGHT', 0.0, 'LEFT ', 0.8088, 'RIGHT']
[0.0, 'RIGHT', 0.0198, 'DOWN ', -0.0, 'RIGHT', 0.0, 'LEFT ']
[0.0002, 'RIGHT', 0.0, 'RIGHT', 0.0, 'DOWN ', 0.0, 'LEFT ']
"""

GAMMA = 0.95
LEARNING_RATE = 0.01
BUFFER_SIZE = 10000

class Net(nn.Module):
    def __init__(self, inputDims, outputDims):
        super(Net, self).__init__()
        self.outputDims = outputDims
        self.inputDims = inputDims
        self.fc1 = nn.Linear(self.inputDims, 32)    #first layer
        self.fc2 = nn.Linear(32, self.outputDims)                #second layer
        #self.fc3 = nn.Linear(16, self.outputDims)   #output layer
        self.device = T.device('cpu')
        self.to(self.device)

    "Implements a feed forward network. state is a one hot vector indicating current state"
    def Forward(self, state):
        x = F.logsigmoid(self.fc1(state))
        #x = F.logsigmoid(self.fc2(x))
        actions = self.fc2(x)
        return actions

class NeuralNetAgent:
    def __init__(self, env, terminalStates):
        self.terminalStates = terminalStates
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.net = Net(self.n_states, self.n_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.successCount = 0
        self.qTable = np.zeros([self.n_states, env.action_space.n])
        self.experienceReplayBuffer = []
        self.erbIndex = 0
        self.epsilon = 1

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
        #epsilon = (1 / np.exp(0.01 * self.successCount)) + 0.01      # e = 0 after 760 successes
        # explore
        if random.random() < self.epsilon:
            return env.action_space.sample()

        # exploit - get best action based on the state
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        return self.EpsilonGreedy(env, state)

    def UpdateModels(self, state, nextState, action, reward):
        if reward == 1:
            self.successCount += 1
            self.epsilon -= 0.002
            if self.epsilon < 0.01:
                self.epsilon = 0.01
        for n in range(1):
            self.Train(state, nextState, action, reward)

    def Train(self, state, nextState, action, reward):
        if len(self.experienceReplayBuffer) < BUFFER_SIZE:
            self.experienceReplayBuffer.append((state, nextState, action, reward))
        else:
            self.experienceReplayBuffer[self.erbIndex] = (state, nextState, action, reward)
        self.erbIndex += 1
        if self.erbIndex >= BUFFER_SIZE:
            self.erbIndex = 0

        for i in range(len(self.recentQs)):
            self.qTable[state][i] = self.recentQs[i]

        #print(f'{state},{action},{nextState},{reward}')
        #state, nextState, action, reward = random.choice(self.experienceReplayBuffer)
        #print(f'{state},{action},{nextState},{reward}')

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
        #for w in self.net.fc1.weight:
        #    print(w)
