# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
import numpy as np

class FlowerDPModel(DPModel): 
    def __init__(self, N=3, c=0.5, prob1=False):
        super().__init__(N=N)
        self.c = c
        self.prob1 = prob1
    def A(self, x, k): # Action space A_k(x)
        return {0, 1, 2}

    def S(self, k): # State space S_k
        return {0, 1, 2}

    def g(self, x, u, w, k): # Cost function g_k(x,u,w)
        if self.prob1:
            return 0
        else:
            return self.c*u + abs(x + u - w)

    def f(self, x, u, w, k): # Dynamics f_k(x,u,w)
        return max(0, min(2, x + u - w ))

    def Pw(self, x, u, k): # Distribution over random disturbances 
        # TODO: 1 lines missing.
        return {0: 0.1, 1: 0.3, 2: 0.6}

    def gN(self, x):
        if self.prob1:
            return -1 if x == 1 else 0
        else:            return 0
    
def a_get_policy(N: int, c: float, x0 : int) -> int:
    # TODO: Code has been removed from here.
    model = FlowerDPModel(N=N, c=c)
    J, pi = DP_stochastic(model)
    return pi[0][x0]

def b_prob_one(N : int, x0 : int) -> float:
    # TODO: Code has been removed from here.
    model = FlowerDPModel(N=N, c=0.5, prob1=True)
    J, pi = DP_stochastic(model)
    return -J[0][x0]


if __name__ == "__main__":
    model = FlowerDPModel()
    pi = [{s: 0 for s in model.S(k)} for k in range(model.N)]
    x0 = 0
    c = 0.5
    N = 3
    print(f"a) The policy choice for {c=} is {a_get_policy(N, c,x0)} should be 1")
    print(f"b) The probability of ending up with a single element in the inventory is {b_prob_one(N, x0)} and should be 0.492")
