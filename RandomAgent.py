#A sample agent that just takes random actions in whatever environment you choose. I've included the 4x4 version and 8x8 versions of Frozen Lake for you to play around with. I've also put CartPole in there because I think it's neat. If you try to activate cartpole, disable the action and observation space print statements.

import gym
import random

class RandomAgent():
    def __init__(self):
        self.ass = 0

    def SuggestMove(self, env):
        return env.action_space.sample()
