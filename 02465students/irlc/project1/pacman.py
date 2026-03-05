# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc import train
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.pacman.gamestate import GameState

east = """ 
%%%%%%%%
% P   .%
%%%%%%%% """ 


east2 = """
%%%%%%%%
%    P.%
%%%%%%%% """

SS2tiny = """
%%%%%%
%.P  %
% GG.%
%%%%%%
"""

SS0tiny = """
%%%%%%
%.P  %
%   .%
%%%%%%
"""

SS1tiny = """
%%%%%%
%.P  %
%  G.%
%%%%%%
"""

datadiscs = """
%%%%%%%
%    .%
%.P%% %
%.   .%
%%%%%%%
"""

# TODO: 30 lines missing.
#raise NotImplementedError("Put your own code here")

def p_next(x : GameState, u: str): 
    """ Given the agent is in GameState x and takes action u, the game will transition to a new state xp.
    The state xp will be random when there are ghosts. This function should return a dictionary of the form

    {..., xp: p, ...}

    of all possible next states xp and their probability -- you need to compute this probability.

    Hints:
        * In the above, xp should be a GameState, and p will be a float. These are generated using the functions in the GameState x.
        * Start simple (zero ghosts). Then make it work with one ghosts, and then finally with any number of ghosts.
        * Remember the ghosts move at random. I.e. if a ghost has 3 available actions, it will choose one with probability 1/3
        * The slightly tricky part is that when there are multiple ghosts, different actions by the individual ghosts may lead to the same final state
        * Check the probabilities sum to 1. This will be your main way of debugging your code and catching issues relating to the previous point.
    """
    # TODO: 8 lines missing.
    #raise NotImplementedError("Return a dictionary {.., xp: p, ..} where xp is a possible next state and p the probability")
    if x.is_won() or x.is_lost():
        return {x: 1.0}
    xp = x.f(u)
    p = {xp: 1.0}

    if xp.player() == 0 or xp.is_won() or xp.is_lost() :
        return p

    while True:

        p_new = {}

        for xg, prob in p.items():
            if xg.is_won() or xg.is_lost():
                p_new[xg] = p_new.get(xg, 0) + prob
                continue
            actions = xg.A()
            prob_a = 1.0 / len(actions)

            for ug in actions:
                xn = xg.f(ug)
                p_new[xn] = p_new.get(xn, 0) + prob * prob_a

        p = p_new

        if all(s.player() == 0 or s.is_won() or s.is_lost() for s in p):
            break
    return p   
def go_east(map): 
    """ Given a map-string map (see examples in the top of this file) that can be solved by only going east, this will return
    a list of states Pacman will traverse. The list it returns should therefore be of the form:

    [s0, s1, s2, ..., sn]

    where each sk is a GameState object, the first element s0 is the start-configuration (corresponding to that in the Map),
    and the last configuration sn is a won GameState obtained by going east.

    Note this function should work independently of the number of required east-actions.

    Hints:
        * Use the GymPacmanEnvironment class. The report description will contain information about how to set it up, as will pacman_demo.py
        * Use this environment to get the first GameState, then use the recommended functions to go east
    """
    # TODO: 5 lines missing.
    #raise NotImplementedError("Return the list of states pacman will traverse if he goes east until he wins the map")
    env = PacmanEnvironment(layout_str=map, render_mode='human')
    x, info = env.reset() # x is a irlc.pacman.gamestate.GameState object. See the online documentation for more examples.
    states = [x]
    while not x.is_won():
        x, _, _, _, _ = env.step('East')
        states.append(x)

    return states

def get_future_states(x, N): 
    # TODO: 4 lines missing.
    #raise NotImplementedError("return a list-of-list of future states [S_0, ... ,S_N]. Each S_k is a state space, i.e. a list of GameState objects.")
    S = []
    S0 = [x]
    S.append(S0)

    for k in range(N):
        #print("k =", k, "|S_k| =", len(S[k]))   # DEBUG
        Sk = S[k]
        Sk_next = set()
        #print (len(Sk), "states in S_k")  # DEBUG
        for x0 in Sk:
            for u0 in x0.A():
                print(len(x0.A()), "actions for state")  # DEBUG
                for x1 in p_next(x0, u0).keys():
                    Sk_next.add(x1)
        #print("k =", k, "|S_{k+1}| =", len(Sk_next))  # DEBUG
        S.append(list(Sk_next))

    return S

def win_probability(map, N=10): 
    """ Assuming you get a reward of -1 on wining (and otherwise zero), the win probability is -J_pi(x_0). """
    # TODO: 5 lines missing.
    #raise NotImplementedError("Return the chance of winning the given map within N steps or less.")
    dp = PacmanWP(map=map, N=N)
    J, policy = DP_stochastic(dp)
    win_probability = -J[0][dp.x]
    return win_probability

def shortest_path(map, N=10): 
    """ If each move has a cost of 1, the shortest path is the path with the lowest cost.
    The actions should be the list of actions taken.
    The states should be a list of states the agent visit. The first should be the initial state and the last
    should be the won state. """
    # TODO: 4 lines missing.
    #raise NotImplementedError("Return the cost of the shortest path, the list of actions taken, and the list of states.")
    dp = PacmanDP(map=map, N=N)
    J, policy = DP_stochastic(dp)

    x = dp.x
    states = [x]
    actions = []

    for k in range(N):
        u = policy[k][x]
        actions.append(u)

        x = x.f(u)
        states.append(x)

        if x.is_won():
            break

    return actions, states


def no_ghosts():
    # Check the pacman_demo.py file for help on the GameState class and how to get started.
    # This function contains examples of calling your functions. However, you should use unitgrade to verify correctness.

    ## Problem 7: Lets try to go East. Run this code to see if the states you return looks sensible.
    states = go_east(east)
    for s in states:
        print(str(s))

    ## Problem 8: try the p_next function for a few empty environments. Does the result look sensible?
    x, _ = PacmanEnvironment(layout_str=east).reset()
    action = x.A()[0]
    print(f"Transitions when taking action {action} in map: 'east'")
    print(x)
    print(p_next(x, action))  # use str(state) to get a nicer representation.

    print(f"Transitions when taking action {action} in map: 'east2'")
    x, _ = PacmanEnvironment(layout_str=east2).reset()
    print(x)
    print(p_next(x, action))

    ## Problem 9
    print(f"Checking states space S_1 for k=1 in SS0tiny:")
    x, _ = PacmanEnvironment(layout_str=SS0tiny).reset()
    states = get_future_states(x, N=10)
    for s in states[1]: # Print all elements in S_1.
        print(s)
    print("States at time k=10, |S_10| =", len(states[10]))

    ## Problem 10
    N = 20  # Planning horizon
    action, states = shortest_path(east, N)
    print("east: Optimal action sequence:", action)

    action, states = shortest_path(datadiscs, N)
    print("datadiscs: Optimal action sequence:", action)

    action, states = shortest_path(SS0tiny, N)
    print("SS0tiny: Optimal action sequence:", action)


def one_ghost():
    # Win probability when planning using a single ghost. Notice this tends to increase with planning depth
    wp = []
    for n in range(10):
        wp.append(win_probability(SS1tiny, N=n))
    print(wp)
    print("One ghost:", win_probability(SS1tiny, N=12))


def two_ghosts():
    # Win probability when planning using two ghosts
    print("Two ghosts:", win_probability(SS2tiny, N=12))

class PacmanDP(DPModel): 
    def __init__(self, map=east, N=20):
        super().__init__(N=N)
        self.map=map
        self.N=N
        self.env = PacmanEnvironment(layout_str=self.map, render_mode=None)
        self.x, info = self.env.reset() # x is a irlc.pacman.gamestate.GameState object. See the online documentation for more examples.
        self.future_states = get_future_states(self.x, self.N)
    def A(self, x, k): # Action space A_k(x)
        return x.A()
    def S(self, k): # State space S_k
        return self.future_states[k]

    def g(self, x, u, w, k): # Cost function g_k(x,u,w)
        if x.is_won():
            return 0
        return 1

    def f(self, x, u, w, k): # Dynamics f_k(x,u,w)
        return w

    def Pw(self, x, u, k): # Distribution over random disturbances 
        # TODO: 1 lines missing.
        return p_next(x, u)

    def gN(self, x):
        return 0

class PacmanWP(DPModel): 
    def __init__(self, map=east, N=20):
        super().__init__(N=N)
        self.map=map
        self.N=N
        self.env = PacmanEnvironment(layout_str=self.map, render_mode=None)
        self.x, info = self.env.reset() # x is a irlc.pacman.gamestate.GameState object. See the online documentation for more examples.
        self.future_states = get_future_states(self.x, self.N)
    def A(self, x, k): # Action space A_k(x)
        return x.A()
    def S(self, k): # State space S_k
        return self.future_states[k]

    def g(self, x, u, w, k): # Cost function g_k(x,u,w)
        return 0

    def f(self, x, u, w, k): # Dynamics f_k(x,u,w)
        return w

    def Pw(self, x, u, k): # Distribution over random disturbances 
        # TODO: 1 lines missing.
        return p_next(x, u)

    def gN(self, x):
        if x.is_won():
            return -1
        return 0

if __name__ == "__main__":
    no_ghosts()
    one_ghost()
    two_ghosts()
