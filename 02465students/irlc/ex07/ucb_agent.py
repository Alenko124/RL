# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex07.simple_agents import BasicAgent
from irlc import savepdf
from irlc import Agent

class UCBAgent(Agent):
    def __init__(self, env, c=2):
        self.c = c
        super().__init__(env)
        self.k = env.action_space.n

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None): 
        # TODO: 2 lines missing.
        #raise NotImplementedError("Train agent here")
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]

    def pi(self, s, k, info=None):
        if k == 0: 
            """ Initialize the agent"""
            # TODO: 3 lines missing.
            #raise NotImplementedError("Reset agent (i.e., make it ready to learn in a new episode with a new optimal action)")
            self.Q = np.zeros(self.k)
            self.N = np.zeros(self.k)
        # TODO: 1 lines missing.
        #raise NotImplementedError("Compute (and return) optimal action")
        t=k+1

        for a in range(self.k):
            if self.N[a] == 0:
                return a
            
        ucb_values = self.Q + self.c * np.sqrt(np.log(t) / (self.N))
        return np.argmax(ucb_values)

    def __str__(self):
        return f"{type(self).__name__}_{self.c}"

from irlc.ex07.bandits import StationaryBandit, eval_and_plot
if __name__ == "__main__":
    r"""Reproduce (SB18, Fig. 2.4) comparing UCB agent to epsilon greedy """
    runs, use_cache = 100, False
    c = 2
    eps = 0.1

    steps = 1000
    env = StationaryBandit(k=10)
    agents = [UCBAgent(env,c=c), BasicAgent(env, epsilon=eps)]
    eval_and_plot(bandit=env, agents=agents, num_episodes=runs, steps=steps, max_episodes=2000, use_cache=use_cache)
    savepdf("UCB_agent")
    plt.show()
