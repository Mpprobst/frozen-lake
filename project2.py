"""
CS660 - HW2
Author: Michael Probst
"""

import gym
import argparse
import numpy as np
import csv
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

ACTION_NAMES = [ 'LEFT ', 'DOWN ', 'RIGHT', 'UP   ' ]

TERMINAL_STATES_4 = [5, 7, 11, 12, 15]
TERMINAL_STATES_8 = [19, 29, 35, 41, 42, 46, 49, 52, 54, 59, 63]

TEST_INDEX = 1000

NUM_TESTS = 50

def Run(agent, env, isTest):
    #Reset will reset the environment to its initial configuration and return that state.
    currentState = env.reset()

    done = False
    stepCount = 0
    #Loop until either the agent finishes or takes 200 actions, whichever comes first.
    while stepCount < 200 and done == False:
        stepCount += 1

        actionToTake = 0
        if isTest:
            actionToTake = agent.GetBestAction(currentState)
        else:
            actionToTake = agent.SuggestMove(env, currentState)

        #Execute actions using the step function. Returns the nextState, reward, a boolean indicating whether this is a terminal state. The final thing it returns is a probability associated with the underlying transition distribution, but we shouldn't need that for this assignment.
        nextState, reward, done, _ = env.step(actionToTake)
        if not isTest:
            agent.UpdateModels(currentState, nextState, actionToTake, reward)

        if args.verbose:
            print(f'Action Taken: {ACTION_NAMES[actionToTake]}')
            #Render visualizes the environment
            env.render()


        currentState = nextState
    return reward
    #endwhile

"""
Main loop for solving the frozen lake problem.
agent is the agent that is currently solving the problem.
size is the size of the lake.
"""
def FrozenLake(agent, size, numEps):
    filename = f'results/{agent}_{size}.csv'
    agentFunc = AGENTS_MAP[agent]
    env = gym.make(f'FrozenLake{LAKE_SIZES[size]}-v0')

    terminalStates = TERMINAL_STATES_4
    if '8x8' == LAKE_SIZES[size]:
        terminalStates = TERMINAL_STATES_8

    agent = agentFunc(env, terminalStates)

    with open(filename, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        if args.verbose:
            #Print out the number of actions and states in the environment (disable for cartpole)
            print(env.action_space.n)
            print(env.observation_space.n)
        for i in range(numEps):

            if i % TEST_INDEX == 0:     # TESTING
                meanReward = 0

                for t in range(NUM_TESTS):
                    #print(f'-----{t}-----\n')
                    value = Run(agent, env, True)
                    meanReward += value
                meanReward /= NUM_TESTS
                print(f'TEST {i / TEST_INDEX}:\t Avg Reward = {meanReward}')
                writer.writerow([i / TEST_INDEX, meanReward])

            else:       # TRAINING
                if args.verbose:
                    print(f'==========\nEPISODE {i}\n==========')
                Run(agent, env, False)

        #endfor
                row = []

    # Shows the action selection policy at the end of training
    if args.verbose:
        print(f'Final Policy:')
        for i in range(len(agent.qTable)):
            row.append(ACTION_NAMES[np.argmax(agent.qTable[i])])
            if (i+1) % 4 == 0:
                print(row)
                row = []
    env.close()

# MAIN
parser = argparse.ArgumentParser(description='Define the agent to solve the frozen lake problem.')
parser.add_argument('--agent', choices=AGENTS_MAP.keys(), default='random', help=' Can be random, dynaQ, qLearn, SARSA, or NN')
parser.add_argument('--size', choices=LAKE_SIZES.keys(), default='4x4', help='Determines size of lake environment. Can be 4x4 or 8x8')
parser.add_argument('--numEpisodes', type=int, default = 10, help='Number of episodes you want the agent to run.')
parser.add_argument('--verbose', help='Print more information.', action='store_true')
args = parser.parse_args()

FrozenLake(args.agent, args.size, args.numEpisodes)
