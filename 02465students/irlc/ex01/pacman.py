from irlc.pacman.pacman_environment import PacmanEnvironment, datadiscs
from irlc import interactive, Agent, train

# Maze layouts can be specified using a string.
layout = """
%%%%%%%%%%
%P.......%
%.%%%%%%.%
%.%    %.%
%.%    %.%
%.%    %.%
%.%    %.%
%.%%%%%%.%
%........%
%%%%%%%%%%
"""
#env = PacmanEnvironment(layout_str=datadiscs, render_mode='human')
env = PacmanEnvironment(layout_str=layout, render_mode='human')
env, agent = interactive(env, Agent(env)) # This makes the environment interactive. Ignore that it needs an Agent for now.
train(env, agent, num_episodes=2)
env.close()