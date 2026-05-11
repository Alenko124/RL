# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex02.inventory import InventoryDPModel
import numpy as np
def a_expected_items_next_day(x : int, u : int) -> float:
    model = InventoryDPModel()
    #expected_number_of_items = None
    # TODO: Code has been removed from here.
    #raise NotImplementedError("Insert your solution and remove this error.")
    expectation = 0
    for w, pw in model.Pw(x, u, 0).items():
        x1 = model.f(x, u, w, 0)
        expectation += pw * x1
    return expectation


def b_evaluate_policy(pi : list, x0 : int) -> float:
    model = InventoryDPModel()
    N = len(pi)

    J = [{} for _ in range(N+1)]
    J[N] = {x: 0 for x in model.S(N)}

    for k in reversed(range(N)):
        for x in model.S(k):

            u = pi[k][x]

            value = 0
            for w, pw in model.Pw(x, u, k).items():

                x_next = model.f(x, u, w, k)
                cost = model.g(x, u, w, k)

                value += pw * (cost + J[k+1][x_next])

            J[k][x] = value

    return J[0][x0]

if __name__ == "__main__":
    model = InventoryDPModel()
    # Create a policy that always buy an item if the inventory is empty.
    pi = [{s: 1 if s == 0 else 0 for s in model.S(k)} for k in range(model.N)]
    x = 0
    u = 1
    x0 = 1
    a_expected_items_next_day(x=0, u=1)
    print(f"Given inventory is {x=} and we buy {u=}, the expected items on day k=1 is {a_expected_items_next_day(x, u)} and should be 0.1")
    print(f"Evaluation of policy is {b_evaluate_policy(pi, x0)} and should be 2.7")
