"""
NeuralNet.py
Author: Michael Probst
Purpose: Implements the Q-Learning model-free RL technique using a neural network to solve the frozen lake problem
Reference: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
"""
import gym
import numpy as np
import random
import tensorflow.compat.v1 as tf
#from tensorflow.python.framework import ops

GAMMA = 0.9
LEARNING_RATE = 0.1

class NeuralNetAgent:
    def __init__(self, env):
        #tf.reset_default_graph()
        tf.disable_eager_execution()
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        #These lines establish the feed-forward part of the network used to choose actions
        self.inputs1 = tf.Variable(tf.zeros([1,self.n_states]))
        self.Qout = tf.Variable(tf.zeros([1,self.n_actions]))#,dtype=tf.float32))
        #self.W = tf.Variable(tf.random_uniform([self.n_states,self.n_actions],0,0.01))
        #self.Qout = tf.matmul(self.inputs1,self.W)
        self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.nextQ = tf.Variable(tf.zeros([1,self.n_actions]))#,dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        self.updateModel = self.trainer.minimize(self.loss)

        self.sess = None
        self.allQ = None

    def GetBestAction(self, state):
        a,self.allQ = self.sess.run([self.predict, self.Qout],feed_dict={self.inputs1:np.identity(16)[state:state+1]})
        return a[0]

    def EpsilonGreedy(self, env, state):
        epsilon = (0.5 / np.exp(0.01 * self.successCount))      # approximately .1 around successCount = 25,000 and e < 0.2 around 6000 This is because I am running 50,000 episodes and the model should be be able to have won half of the time by the end of training
        # explore
        if random.random() < epsilon:
            return env.action_space.sample()

        # exploit - get best action based on the state
        return self.GetBestAction(state)

    def SuggestMove(self, env, state):
        return EpsilonGreedy(env, state)

    def UpdateModels(self, state, nextState, action, reward):
        #Obtain the Q' values by feeding the new state through our network
        Q1 = self.sess.run(self.Qout,feed_dict={self.inputs1:np.identity()[nextState:nextState+1]})
        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q1)
        targetQ = self.allQ
        targetQ[0,action] = reward + GAMMA * maxQ1
        #Train our network using target and predicted Q values
        _,W1 = self.sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[state:state+1],nextQ:targetQ})
