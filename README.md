# Frozen Lake
Author: Michael Probst
Date: 10/23/2020

## Overview
This repository contains a solution to Problem Set 2 for CS660 - Sequential Decision Making. The purpose of this project was to solve the open ai gym Frozen Lake problem using various strategies of reinforcement learning.

Files include: project2.py, DynaQ.py, QLearning.py, SARSA.py, NeuralNet.py, RandomAgent.py, and writeup.pdf

## Runing This Program
The file that solves the frozen lake problem is project2.py which utilizes various files based on which technique is used to solve the problem. 

To run the program, simply enter `python project2.py` 
However, several other arguements can be passed to grant more control over the program which are: --agent, --size, --numEpisodes, and --verbose

For more information on each of the arguments, enter `python project2.py --help`  

## Implementation Decisions
The structure of this program was inspired by Problem Set 1 in which there is one main file that executes the simulation (project2.py), and the other files describe various agents that can solve the frozen lake problem. The agents do so by recommending an action which is always implemented in the SuggestMove() method in each agent class.


