"""
CS660 - HW2
Author: Michael Probst
"""

import gym
import argparse
import numpy as np
from RandomAgent import RandomAgent
from DynaQ import DynaQAgent
from QLearning import QLearningAgent
from SARSA import SARSAAgent
from NeuralNet import NeuralNetAgent

AGENTS_MAP = {'random' : RandomAgent,
               'dynaQ' : DynaQAgent,
               'qLearn': QLearningAgent,
               'SARSA': SARSAAgent,
               'NN' : NeuralNetAgent  }

LAKE_SIZES = {'4x4' : '',
              '4X4' : '',
              '8x8' : '8x8',
              '8X8' : '8x8' }

ACTION_NAMES = [ 'LEFT', 'DOWN', 'RIGHT', 'UP' ]

"""
Main loop for solving the frozen lake problem.
agent is the agent that is currently solving the problem.
size is the size of the lake.
"""
def FrozenLake(agent, size, numEps):
    agentFunc = AGENTS_MAP[agent]
    env = gym.make(f'FrozenLake{LAKE_SIZES[size]}-v0')
    agent = agentFunc(env)

    if args.verbose:
        #Print out the number of actions and states in the environment (disable for cartpole)
        print(env.action_space.n)
        print(env.observation_space.n)

    for i in range(numEps):
        #Reset will reset the environment to its initial configuration and return that state.
        currentState = env.reset()

        done = False
        stepCount = 0
        if i % 100 == 0:
            print(f'EPISODE {i}')
        if args.verbose:
            print(f'==========\nEPISODE {i}\n==========')
        #Loop until either the agent finishes or takes 200 actions, whichever comes first.
        while stepCount < 200 and done == False:
            stepCount += 1

            actionToTake = agent.SuggestMove(env, currentState)

            #Execute actions using the step function. Returns the nextState, reward, a boolean indicating whether this is a terminal state. The final thing it returns is a probability associated with the underlying transition distribution, but we shouldn't need that for this assignment.
            nextState, reward, done, _ = env.step(actionToTake)
            if reward == 1:
                print("win!")

            agent.UpdateModels(currentState, nextState, actionToTake, reward)

            if args.verbose:
                print(f'Action Taken: {ACTION_NAMES[actionToTake]}')
                #Render visualizes the environment
                env.render()

            currentState = nextState
        #endwhile
        

    env.close()


# MAIN
parser = argparse.ArgumentParser(description='Define the agent to solve the frozen lake problem.')
parser.add_argument('--agent', choices=AGENTS_MAP.keys(), default='random', help=' Can be random, dynaQ, qLearn, SARSA, or NN')
parser.add_argument('--size', choices=LAKE_SIZES.keys(), default='4x4', help='Determines size of lake environment. Can be 4x4 or 8x8')
parser.add_argument('--numEpisodes', type=int, default = 10, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Print more information.', action='store_true')
args = parser.parse_args()

FrozenLake(args.agent, args.size, args.numEpisodes)
