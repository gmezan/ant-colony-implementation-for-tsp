import numpy as np
from .Ant import Ant
"""
We consider the problem where n items should be packed in m knapsacks
 of distinct capacities ci, i = 1, . . . , m. Each item j has an associated 
 value pj and weight wj, and the problem is to select m disjoined subsets of
 items, such that subset i fits into capacity ci and the total profit of 
 the se- lected items is maximized.
"""

TINY_VAL = 0.0001
LONG_VAL = 10000

# Multiple Knapsack
"""
p: value of items: p_j, j = {0...n-1}
c: capacities of knapsacks: c_i, i = {0...m-1}
w: weights of items: w_j
"""
class AcoMkp:
    def __init__(self, p: list, c: list, w: list, n_ants: int, max_iter = 1000000, alpha = 0.5, beta = 1, rho = 0.3) -> None:
        # Number of items: p_j = {0,1,2,..., n-1}
        self.__N = len(p)
        # Number of knapsacks: c_i = {0,1,2,..., m-1}
        
        self.__M = len(c)
        # Number of ants
        self.__N_ANTS = n_ants

        # Profit vector p[n]: pj of profit for item j
        self.p = np.array(p)
        # Weights vector w[n]: item j consumes wj units of capacity c_i
        self.w = np.array(w)
        # Constraints (capacities) c[m] vector for resource? c_i
        self.c = np.array(c)
        # solution S_k (t) or "x" is x[m][n] 

        # real pheromone[n][m] % pheromone matrix (pheromone trail tau)
        self.pheromone = np.random.rand((self.__M, self.__N));
        # single_ant ant[m] % structure of type single_ant
        self.ants = [Ant(self.__M, self.__N) for _ in range(n_ants) ]

        # Hyper parameters:
        self.alpha = alpha  # weight for trails
        self.beta = beta    # weight for lengths (weights) (recommended beta > alpha)
        self.rho = rho      # trail/pheromone evaporation [0, 1]
        self.q_o = 0 #np.random.rand()
        self.max_iter = max_iter;
        self.tau_o = 0.005

        # Parameters
        self.terminate = False;
        self.counter = 0
        self.best_fit = None
        self.best_fit_profit = 0.0

        self.pheromones = []

    
    def init_random_pheromone_trails(self):
        self.pheromone = np.random.rand((self.__M, self.__N))
        self.pheromones.append(self.pheromone.tolist())

    # x is the selection S^~ _k ( t )
    # u_i(k,t)
    def compute_consumption_mu(self, x):
        return np.matmul(x, self.w)

    # y_i(k,t)
    def compute_remaining_gamma(self, x):
        return self.c - self.compute_consumption_mu(x)

    # ∂_ij(k,t)
    def compute_tightness_delta(self, x):
        return self.w / self.compute_remaining_gamma(x)[:,None]

    def compute_avg_tightness_delta(self, x):
        return self.compute_tightness_delta(x).mean(axis = 0)

    # compute pseudo utility: local heuristic
    def compute_pseudo_utility_eta(self, x):
        return self.p / self.compute_avg_tightness_delta(x)

    def are_ants_done(self):
        return np.all(np.array([ant.done for ant in self.ants], dtype=bool) == True)

    # add item at ant ants[k]
    def add_item(self, k: int):
        if not np.any(self.calculate_allowed(k)):
            self.ants[k].done = True
            return
        
        ant_r = self.ants[k]

        if not np.any(ant_r.s):
            # VARIATION choose random for first pick
            self.ants[k].add(np.random.choice(np.where(self.ants[k].allowed==True)[0]))
            return
        
        aux_1 = (ant_r.allowed * self.pheromone) ** self.alpha  # tau_i ^ alpha
        aux_2 = (ant_r.allowed * ((1.0 * self.p) / self.compute_avg_tightness_delta(ant_r.s))) ** self.beta  # eta_i ^ beta

        # P^k _i_p (t)
        probs = (aux_1 * aux_2) / np.dot(aux_1, aux_2)
        probs /= np.linalg.norm(probs, ord=1)
        
        # applying meta heuristic with q_o 
        probs[probs < np.random.rand()] = LONG_VAL

        # move the ant using meta heuristic
        self.ants[k].add(np.argmin(probs))

    # Calculate allowed items for ant ants[k]
    def calculate_allowed(self, k: int):
        ant_r = self.ants[k]
        # check with tabu and s_k solution
        init_allowed = np.ones(self.__N) * (ant_r.tabu_list == False) * (self.ants[k].s == False)
        constraint_filter = np.zeros((self.__N,), dtype=bool)
        aux_x = np.zeros(self.__N)
        # take only the elements that satisfy all the constraints
        for idx in np.where(init_allowed == True)[0]:
            aux_x.fill(0)
            aux_x[idx] = 1
            constraint_filter[idx] = np.all(self.compute_remaining_gamma(ant_r.s + aux_x) >= 0)

        self.ants[k].allowed = constraint_filter * init_allowed
        return self.ants[k].allowed

    def start_ants(self):
        for k in range(self.__N_ANTS):
            self.ants[k].reset()

    def update_objective_func(self, k: int):
        fit = np.dot(self.ants[k].s, self.p)
        # G(L_k) = Q * L_k
        self.ants[k].profit = (1.0 * fit) / np.sum(self.p)
        
        # VARIATION with >=
        if fit >= self.best_fit_profit:
            #print(str(self.ants[k].profit) + " > " + str(self.best_fit_profit))
            self.best_fit = self.ants[k].s.tolist()
            self.best_fit_profit = fit
            # VARIATION to make the best profit more valuable
            self.pheromone *= (1-self.rho)
            self.pheromone += self.ants[k].profit * self.ants[k].s

    # update pheromone
    def update_trails(self):
        delta_tau = np.zeros(self.__N)
        for ant in self.ants:
            delta_tau += ant.profit * ant.s

        # Variation to make each delta of ant counts less
        #delta_tau /= self.__N_ANTS

        self.pheromone *= (1-self.rho)
        self.pheromone += delta_tau
        self.pheromones.append(self.pheromone.tolist())