# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
import numpy as np

class Toy2DControl(ControlModel):
    def get_cost(self):
        # You get the cost-function for free because it can be anything as far as this problem is concerned.
        return SymbolicQRCost(Q=np.eye(2), R=np.eye(1))

    # TODO: 2 lines missing.
    #raise NotImplementedError("Insert your solution and remove this error.")
    def sym_f(self, x: list, u: list, t=None): 
        r""" Return a symbolic expression representing the Kuramoto model.
        The inputs x, u are themselves *lists* of symbolic variables (insert breakpoint and check their value).
        you have to use them to create a symbolic object representing f, and return it as a list. That is, you are going to return

        .. codeblock:: python

            return [f_val]

        where ``f_val`` is the symbolic expression corresponding to the dynamics, i.e. :math:`u(t) + \cos( x(t))`.
        Note you can use trigonometric functions like ``sym.cos``.
        """
        first=x[1]
        second=sym.cos(u[0]+x[0])
        return [first, second]  
def toy_simulation(u0 : float, T : float) -> float:
    # TODO: 4 lines missing.
    model= Toy2DControl()
    x0 = np.asarray([np.pi/2, 0])
    xs, us, tt, _ = model.simulate(x0, u0, t0=0, tF=T)

    return xs[-1][0]

if __name__ == "__main__":
    x0 = np.asarray([np.pi/2, 0])
    wT = toy_simulation(u0=0.4, T=5)
    print(f"Starting in x0=[pi/2, 0], after T=5 seconds the system is an an angle {wT=} (should be 1.265)") 
