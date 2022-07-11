import numpy as np
from .Ant import Ant
"""

IMPLEMENTATION OF: A New Version of Ant System for Subset Problems

maximize Cj”=, pjxj
subject to CY=, rijxj <= ci i = 1, ..., m, 
xj E {0,1} j = 1, ... n
"""

TINY_VAL = 0.0001
LONG_VAL = 10000

# One Knapsack
class AsSubset:
    def __init__(self, p: list, c: list, r: list, n_ants: int, max_iter = 1000000, alpha = 0.5, beta = 1, rho = 0.3) -> None:
        # Number of items: j = {0,1,2,..., n-1}
        self.__N = len(p)
        # Number of constraints: i = {0,1,2,..., m-1}
        self.__M = len(c)
        # Number of ants
        self.__N_ANTS = n_ants

        # Profit vector: pj of profit for item j
        self.p = np.array(p)
        # Resources matrix: item j consumes rij units of resource i
        self.r = np.array(r)
        # Constraints matrix for resource i
        self.c = np.array(c)

        # real pheromone[n][n] % pheromone matrix (pheromone trail tau)
        self.pheromone = None;
        # single_ant ant[m] % structure of type single_ant
        self.ants = [Ant(self.__N) for _ in range(n_ants) ]

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

        # The amount of resource i consumed at the time t, ant k 
        self.mu = np.zeros(self.__M)
        # x_j € {0, 1} j = 1...m
        self.x = np.zeros(self.__N)

        self.pheromones = []

    
    def init_random_pheromone_trails(self):
        self.pheromone = np.random.rand(self.__N)
        self.pheromones.append(self.pheromone.tolist())

    # x is the selection S^~ _k ( t )
    # u_i(k,t)
    def compute_consumption_mu(self, x):
        return np.matmul(self.r, x)

    # y_i(k,t)
    def compute_remaining_gamma(self, x):
        return self.c - self.compute_consumption_mu(x)

    # ∂_ij(k,t)
    def compute_tightness_delta(self, x):
        return self.r / self.compute_remaining_gamma(x)[:,None]

    def compute_avg_tightness_delta(self, x):
        return self.compute_tightness_delta(x).mean(axis = 0)

    # compute pseudo utility: local heuristic
    def compute_pseudo_utility_eta(self, x):
        return self.p / self.compute_avg_tightness_delta(x)

    def is_terminate(self) -> bool:
        if self.terminate:
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.max_iter:
            self.counter = 0
            self.terminate = True
            return True

        return self.terminate

    def are_ants_done(self):
        return np.all(np.array([ant.done for ant in self.ants], dtype=bool) == True)

    # add item at ant ants[k]
    def add_item(self, k: int) -> bool:
        x = None
        if not np.any(self.calculate_allowed(k)):
            self.ants[k].done = True
            return False
        
        ant_r = self.ants[k]

        if not np.any(ant_r.s):
            #print(ant_r.allowed)
            #eta_aux = self.p / self.compute_avg_tightness_delta(ant_r.s)
            # VARIATION choose random for first pick
            x = np.random.choice(np.where(self.ants[k].allowed==True)[0])
            self.ants[k].add(x)
            return True
        else:
            eta_aux = np.ones(len(self.p)) * np.random.rand()

        aux_1 = (ant_r.allowed * self.pheromone) ** self.alpha  # tau_i ^ alpha
        aux_2 = (ant_r.allowed * eta_aux) ** self.beta  # eta_i ^ beta

        # P^k _i_p (t)
        probs = aux_1 * aux_2 / np.dot(aux_1, aux_2)
        probs = probs / np.sum(probs)
        
        # applying meta heuristic with q_o 
        rand = np.random.rand()
        aux = ( probs > rand) * probs
        if (rand <= self.q_o) or np.all(aux == 0):
            x = np.argmax(probs)
        else:
            aux[aux <= 0.0] = LONG_VAL
            x = np.argmin(aux)

        # move the ant using meta heuristic
        self.ants[k].add(x)
        return True

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
        self.ants[k].profit = (1.0 * fit)# / np.sum(self.p)
        
        # VARIATION with >=
        if self.ants[k].profit >= self.best_fit_profit:
            #print(self.ants[k].s)
            #print(self.p)
            #print(self.ants[k].profit)
            self.best_fit = self.ants[k].s
            self.best_fit_profit = fit
            # VARIATION to make the best profit more valuable
            self.pheromone = (1-self.rho) * self.pheromone + self.rho* self.ants[k].profit * self.ants[k].s

    # update pheromone
    def update_trails(self):
        delta_tau = np.zeros(self.__N)
        for ant in self.ants:
            delta_tau += ant.profit * ant.s

        # Variation to make each delta of ant counts less
        delta_tau /= self.__N_ANTS

        self.pheromone = (1-self.rho) * self.pheromone + self.rho* delta_tau
        self.pheromones.append(self.pheromone.tolist())