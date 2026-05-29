from irlc.exam.exam2025spring.inventory import InventoryDPModel
from irlc.exam.exam2025spring.dp import DP_stochastic
from irlc.ex02.deterministic_inventory import DeterministicInventoryDPModel
from irlc.ex02.dp_model import DPModel

# TODO: Code has been removed from here. 

def a_expected_cost(x0 : int, u0 : int) -> float:
    # TODO: Code has been removed from here.
    #raise NotImplementedError("Insert your solution and remove this error.")
    expected_cost = 0.1*(u0+(x0+u0)**2)+0.7*(u0+(x0+u0-1)**2)+0.2*(u0+(x0+u0-2)**2)
    return expected_cost

class CakeDPModel(DPModel): 
    def __init__(self, N=3, cost_per_cake=0.5):
        super().__init__(N=N)
        self.cost_per_cake = cost_per_cake

    def A(self, x, k): # Action space A_k(x)
        return {0, 1, 2}

    def S(self, k): # State space S_k
        return {0, 1, 2}

    def g(self, x, u, w, k): # Cost function g_k(x,u,w)
        return self.cost_per_cake * u - min(x + u, w)
    
    def f(self, x, u, w, k): # Dynamics f_k(x,u,w)
        return max(0, min(2, x + u - w ))

    def Pw(self, x, u, k): # Distribution over random disturbances 
        # TODO: 1 lines missing.
        return {0: 0.1, 1: 0.7, 2: 0.2}

    def gN(self, x):
        return x*x
    

class CakeLazyDPModel(DPModel): 
    def __init__(self, N=3, cost_per_cake=0.5):
        super().__init__(N=N)
        self.cost_per_cake = cost_per_cake

    def A(self, x, k): # Action space A_k(x)
        if k%2 == 0:
            return {2}
        else:
            return {0}

    def S(self, k): # State space S_k
        return {0, 1, 2}

    def g(self, x, u, w, k): # Cost function g_k(x,u,w)
        return self.cost_per_cake * u - min(x + u, w)
    
    def gN(self, x) -> float:
        return x*x
    def f(self, x, u, w, k): # Dynamics f_k(x,u,w)
        return max(0, min(2, x + u - w ))

    def Pw(self, x, u, k): # Distribution over random disturbances 
        # TODO: 1 lines missing.
        return {0: 0.1, 1: 0.7, 2: 0.2}

    def gN(self, x):
        return x*x

def b_best_action(N : int, cost_per_cake : float, k : int, x : int) -> int:
    # TODO: Code has been removed from here.
    #raise NotImplementedError("Insert your solution and remove this error.")
    model = CakeDPModel(N=N, cost_per_cake=cost_per_cake)
    J, pi = DP_stochastic(model)
    best_action = pi[k][x]

    return best_action

def c_lazy_baker(N : int, cost_per_cake : float, x0 : int) -> float:
    # TODO: Code has been removed from here.
    #raise NotImplementedError("Insert your solution and remove this error.")
    model = CakeLazyDPModel(N=N, cost_per_cake=cost_per_cake)
    J, pi = DP_stochastic(model)
    cost = J[0][x0]
    return cost

if __name__ == "__main__":
    print(f"a) The expected cost should be 1.3 and you got {a_expected_cost(x0=0, u0=1)=}")
    print(f"b) Using the modified cost the best action is 1 and you got: {b_best_action(N=3, cost_per_cake=0.8, k=0, x=1)=}")
    print(f"c) The expected cost for the lazy baker is approximately 1.311 and you got: {c_lazy_baker(N=3, cost_per_cake=0.7, x0=0)=}")
