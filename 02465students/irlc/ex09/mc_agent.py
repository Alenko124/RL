# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
import matplotlib.pyplot as plt
from irlc.ex08.rl_agent import TabularAgent
from irlc import main_plot, savepdf, train
from irlc import interactive


def get_MC_return_SA(episode, gamma, first_visit=True):
    """ Helper method for computing the MC returns.
    Given an episodes in the form [ (s0,a0,r1), (s1,a1,r2), ...]
    this function computes (if first_visit=True) a new list

    > [((s,a), G) , ... ]

    consisting of the unique $(s_t,a_t)$ pairs in episode along with their return G_t (computed from their first occurance).
    Alternatively, if first_visit=False, the method return a list of same length of episode
    with all (s,a) pairs and their return.
    """
    ss = [s for s, _, _ in episode]   # gdje je s zapravo (s,a)
    G = 0
    returns = []

    for t in reversed(range(len(episode))):
        s_t, _, r_tp1 = episode[t]

        # Compute return G_t
        G = gamma * G + r_tp1

        # First-visit or every-visit condition
        if not first_visit or s_t not in ss[:t]:
            returns.append((s_t, G))

    # Optional: reverse to get chronological order
    returns.reverse()

    return returns

class MCAgent(TabularAgent): 
    def __init__(self, env, gamma=1.0, epsilon=0.05, alpha=None, first_visit=True):
        if alpha is None:
            self.returns_sum_S = defaultdict(float)
            self.returns_count_N = defaultdict(float)
        self.alpha = alpha
        self.first_visit = first_visit
        self.episode = []
        super().__init__(env, gamma, epsilon) 

    def pi(self, s,k, info=None): 
        """
        Compute the policy of the MC agent. Remember the agent is epsilon-greedy. You can use the pi_eps(s,info)-function defined
        in the TabularAgent class.
        """
        # TODO: 1 lines missing.
        #raise NotImplementedError("Compute action here using the Q-values. (remember to be epsilon-greedy)")
        return self.pi_eps(s, info)  # koristi TabularAgent helper

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):  
        """
        Consult your implementation of value estimation agent for ideas. Note you can index the Q-values as

        >> self.Q[s, a] = new_q_value

        see comments in the Agent class for more details, however for now you can consider them as simply a nested
        structure where ``self.Q[s, a]`` defaults to 0 unless the Q-value has been updated.
        """
        # TODO: 12 lines missing.
        #raise NotImplementedError("Train the agent here.")
        # 1) skupljaj epizodu
        self.episode.append((s, a, r))

        # 2) treniraj tek kad epizoda završi
        if done:
            # vrati [( (s,a), G ), ...]
            sa_returns = get_MC_return_SA(
                [( (s,a), None, r ) for (s,a,r) in self.episode],
                self.gamma,
                self.first_visit
            )

            for sa, G in sa_returns:
                s_t, a_t = sa

                if self.alpha is not None:
                    # incremental update
                    self.Q[s_t, a_t] += self.alpha * (G - self.Q[s_t, a_t])
                else:
                    # prosjek returna
                    self.returns_sum_S[(s_t, a_t)] += G
                    self.returns_count_N[(s_t, a_t)] += 1
                    self.Q[s_t, a_t] = (
                        self.returns_sum_S[(s_t, a_t)] /
                        self.returns_count_N[(s_t, a_t)]
                    )

            # 3) reset epizode
            self.episode = []

    def __str__(self):
        return f"MC_{self.gamma}_{self.epsilon}_{self.alpha}_{self.first_visit}"

if __name__ == "__main__":
    """ Load environment but make sure it is time-limited. Can you tell why? """
    envn = "SmallGridworld-v0"

    from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment, BookGridEnvironment
    env = SuttonCornerGridEnvironment(uniform_initial_state=True)
    # env = BookGridEnvironment(living_reward=-0.05) # Uncomment to test an alternative environment with a negative living reward.

    gamma = 1 
    episodes = 20000
    experiment="experiments/mcagent_smallgrid"
    agent = MCAgent(env, gamma=gamma, first_visit=True)
    train(env, agent, experiment_name=experiment, num_episodes=episodes, return_trajectory=False)
    main_plot(experiments=[experiment], resample_ticks=200) 
    plt.title("Smallgrid MC agent value function")
    plt.ylim([-10, 0])
    savepdf("mcagent_smallgrid") 
    plt.show() 

    env, agent = interactive(env, agent)
    env.reset()
    env.plot()
    plt.title(f"MC on-policy control of {envn} using first-visit")
    savepdf("MC_agent_value_smallgrid")
    plt.show(block=False)
