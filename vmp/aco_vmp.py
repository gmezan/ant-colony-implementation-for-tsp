from http import server
import numpy as np
from .Ant import Ant
"""
We consider the problem where n items should be packed in m knapsacks
 of distinct capacities ci, i = 1, . . . , m. Each item j has an associated 
 value pj and weight wj, and the problem is to select m disjoined subsets of
 items, such that subset i fits into capacity ci and the total profit of 
 the se- lected items is maximized.
"""

TINY_VAL = 0.001
LONG_VAL = 10000

# Multiple Knapsack
"""
p: value of items: p_j, j = {0...n-1}
c: capacities of knapsacks: c_i, i = {0...m-1}
w: weights of items: w_j
"""

"""

j = {1...N} # n vms
i = {1...M} # m pms
l = {1...D} # d constraints each  


capcities (servers):  c[i][l], pm c[i] has l constraints

solution/selection:   x[i][j], x=1 when vm j placed in pm i

profit:             p[j], 0 = dont place. 1 = best_effort, 2 = HPC

weights:            w[j][l], vm j has w[j], a vector of size l with its constraints 

oc: overcommitting for each dimension and for each server
    oc[i][l] 

"""
class AcoVmp:
    def __init__(self, p: list, c: list, w: list, oc: list, n_ants: int, max_iter = 1000000, alpha = 0.5, beta = 1, rho = 0.3, kp_first = True, w1=1, w2=1, w3=0.5, tau_max = 20, tau_min = 0.1) -> None:
        # Number of items: p_j = {0,1,2,..., n-1}
        self.__N = len(p)
        # Number of knapsacks: c_i = {0,1,2,..., m-1}
        self.__M = len(c)
        # Number of constraints
        self.__D = len(c[0])



        # Number of ants
        self.__N_ANTS = n_ants

        # Profit vector p[n]: pj of profit for item j
        self.p = np.array(p)
        # Weights vector w[j][l]: item j consumes wj units of capacity c_i
        self.w = np.array(w)
        # Constraints (capacities) c[i][l] vector for resource? c_i
        self.base_c = np.array(c)
        self.oc = np.array(oc)
        assert self.oc.shape == self.base_c.shape
        self.c = self.base_c * self.oc
        assert self.c.shape == self.base_c.shape

        # solution S_k (t) or "x" is x[i][j] 

        # real pheromone[i][j] % pheromone matrix (pheromone trail tau)
        self.pheromone = np.random.rand(self.__M, self.__N);
        # single_ant ant[m] % structure of type single_ant
        self.ants = [Ant(self.__M, self.__N) for _ in range(n_ants) ]

        # Hyper parameters:
        self.alpha = alpha  # weight for trails
        self.beta = beta    # weight for lengths (weights) (recommended beta > alpha)
        self.rho = rho      # trail/pheromone evaporation [0, 1]
        self.q_o = 0 #np.random.rand()
        self.max_iter = max_iter;
        self.tau_o = 0.005
        self.kp_first = kp_first

        self.tau_max = tau_max
        self.tau_min = tau_min

        # Parameters
        self.terminate = False;
        self.counter = 0
        self.best_fit = None
        self.best_fit_profit = 0.0

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # Transients
        #self.pheromones = []
        self.init_allowed = np.ones((self.__M, self.__N))

    
    def init_random_pheromone_trails(self):
        self.pheromone = np.random.rand(self.__M, self.__N)
        #self.pheromones.append(self.pheromone.tolist())

    # must return a [i][l] matrix containing current resources being used
    def compute_consumption_mu(self, x):
        return np.matmul(x, self.w)

    # y_i(k,t)
    def compute_remaining_gamma(self, x):
        return self.c - self.compute_consumption_mu(x)

    # ∂_ij(k,t)
    # 3 dimensional matrix VMsxPMs
    def compute_tightness_delta(self, x):
        ans = []
        for k_remaining in self.compute_remaining_gamma(x):
            k_remaining[k_remaining <= 0] = TINY_VAL
            ans.append(self.w / k_remaining)
        
        return np.array(ans)

    def compute_avg_tightness_delta(self, x):
        return self.compute_tightness_delta(x).mean(axis = 2)

    # compute pseudo utility: local heuristic
    def compute_pseudo_utility_eta(self, x):
        return (self.p ** self.w3) * self.compute_obj(x) / self.compute_avg_tightness_delta(x)

    def are_ants_done(self):
        return np.all(np.array([ant.done for ant in self.ants], dtype=bool) == True)

    # add item at ant ants[k]
    def add_item(self, k: int):
        if not np.any(self.calculate_allowed(k)):
            self.ants[k].done = True
            return
        ant_r = self.ants[k]

        # pick random first VM j for first PM i
        if not np.any(ant_r.s):
            rc = np.transpose(np.where(ant_r.allowed == True))
            [i,j] = rc[np.random.choice(len(rc))]
            # VARIATION choose random for first pick
            self.ants[k].add(i,j)
            return

        """
            Step: is to choose a item to be placed or knasack where to be placed, then 
            calculate the probabilities targeting the next kanpsack where to be placed or nec item to be place
        """

        #avg_tightness = self.compute_avg_tightness_delta(ant_r.s)
        
        local_heuristic = self.compute_pseudo_utility_eta(ant_r.s)

        aux_1 = None
        aux_2 = None
        probs = None

        if self.kp_first:
            #j = self.choose_next_item(ant_r.allowed, avg_tightness=avg_tightness)
            i = self.choose_next_knapsack(ant_r.allowed, local_heuristic=local_heuristic)

            # MxN, aux_2 "Profit" based vector TODO: update with functions
            aux_1 = (ant_r.allowed[i] * self.pheromone[i]) ** self.alpha  # tau_i ^ alpha
            aux_2 = (ant_r.allowed[i] * local_heuristic[i]) ** self.beta  # eta_i ^ beta

            # P^k _i_p (t)
            probs = aux_1 * aux_2
            probs[probs <= 0] = TINY_VAL
            probs /= np.linalg.norm(probs, ord=1)
            
            # applying meta heuristic with q_o: choose vm
            # move the ant using meta heuristic
            j = (np.random.rand() < probs.cumsum()).argmax()
            if self.ants[k].allowed[i][j]:
                self.ants[k].add(i, j)
        
        else:
            j = self.choose_next_item(ant_r.allowed, local_heuristic=local_heuristic)
            aux_1 = (ant_r.allowed[:,j] * self.pheromone[:,j]) ** self.alpha  # tau_i ^ alpha
            aux_2 = (ant_r.allowed[:,j] * local_heuristic[:,j]) ** self.beta  # eta_i ^ beta

            # P^k _i_p (t)
            probs = aux_1 * aux_2
            probs[probs <= 0] = TINY_VAL
            probs /= np.linalg.norm(probs, ord=1)
            
            # applying meta heuristic with q_o: choose vm
            # move the ant using meta heuristic
            i = (np.random.rand() < probs.cumsum()).argmax()
            if self.ants[k].allowed[i][j]:
                self.ants[k].add(i, j)

        del ant_r, probs, aux_1, aux_2

    def choose_next_item(self, allowed, local_heuristic):
        item_chooser = (allowed * local_heuristic).mean(axis = 0)
        item_chooser /= np.linalg.norm(item_chooser, ord=1)
        return (np.random.rand() < item_chooser.cumsum()).argmax()

    def choose_next_knapsack(self, allowed, local_heuristic):
        item_chooser = (self.pheromone * allowed * local_heuristic).mean(axis = 1)
        item_chooser /= np.linalg.norm(item_chooser, ord=1)
        return (np.random.rand() < item_chooser.cumsum()).argmax()

    # Calculate allowed items for ant ants[k]
    def calculate_allowed(self, k: int):
        ant_r = self.ants[k]
        self.init_allowed.fill(1)
        # check with tabu and s_k solution
        self.init_allowed *= (ant_r.tabu_list == False) * (ant_r.s == False) * 1
        self.ants[k].allowed.fill(0)

        aux_mem = []
        for [i,j] in np.transpose(np.where(self.init_allowed == False)):
            if j in aux_mem:
                continue
            else:
                self.init_allowed[:,j] = 0
                aux_mem.append(j)
        
        # take only the elements that satisfy all the constraints
        for [i,j] in np.transpose(np.where(self.init_allowed == True)):
            ant_r.s[i][j] = 1
            self.ants[k].allowed[i][j] = np.all(self.compute_remaining_gamma(ant_r.s) > 0)
            ant_r.s[i][j] = 0

        del ant_r, aux_mem
        return self.ants[k].allowed

    def start_ants(self):
        for k in range(self.__N_ANTS):
            self.ants[k].reset()

    def update_objective_func(self, k: int):
        # TODO: update the objective function
        #fit = np.dot(self.ants[k].s, self.p).sum()

        fit = self.compute_obj(self.ants[k].s) # np.dot(self.ants[k].s, self.co(self.ants[k].s)).sum() / np.sum(self.p)
        # G(L_k) = Q * L_k
        self.ants[k].profit = fit

        if self.best_fit_profit == None:
            self.best_fit_profit = fit

        # VARIATION with >=
        if fit >= self.best_fit_profit:
            #print(str(self.ants[k].profit) + " > " + str(self.best_fit_profit))
            self.best_fit = self.ants[k].s.tolist()
            self.best_fit_profit = fit
            # VARIATION to make the best profit more valuable
            self.pheromone *= (1-self.rho)
            self.pheromone += self.ants[k].profit * self.ants[k].s

            self.apply_tau_max_min()
        
        del fit

    # update pheromone
    def update_trails(self):
        delta_tau = np.zeros((self.__M, self.__N))
        for ant in self.ants:
            delta_tau += ant.profit * ant.s

        # Variation to make each delta of ant counts less
        #delta_tau /= self.__N_ANTS

        self.pheromone *= (1-self.rho)
        self.pheromone += delta_tau/self.__N_ANTS
        #self.pheromones.append(self.pheromone.tolist())
        del delta_tau

        self.apply_tau_max_min()

    """
    Parameters: 
        - x: current solution of ant, ants[k].s
        
    Returns 
        - L^k_{ij}(ants[k].s): loss function of current ant solution and for each (i,j), shape=MxN
    """

    def compute_obj(self, x):
        obj =  np.matmul(x, self.p).sum() / ( self.p.sum() * (1 + ((self.interference_hpc(x) ** self.w1) * (self.cpu_oc_hpc(x) * self.ram_oc_hpc(x)) ** self.w2) ))
        #print(obj)
        return obj

    def interference_hpc(self, x):
        # find servers with HPC vms
        p_hpc_vms = np.where(self.p >= 2)[0]

        #num_cpus_hpc_per_server_per_vm = np.zeros(x.shape)
        #num_cpus_hpc_per_server_per_vm[:,p_hpc_vms] = x[:,p_hpc_vms]
        #num_cpus_hpc_per_server_per_vm *= self.w[:,0]

        # compute total interference in vm
        # p * numCPUs; Assuming l=0: num CPUs 
        num_p_cpus_per_vm_per_server = x * self.w[:,0] * self.p
        num_p_cpus_HPC_per_vm_per_server = np.zeros(x.shape)
        num_p_cpus_HPC_per_vm_per_server[:,p_hpc_vms] = num_p_cpus_per_vm_per_server[:,p_hpc_vms]
        
        num_p_cpus_per_server = num_p_cpus_per_vm_per_server.sum(axis = 1)

        aux = num_p_cpus_HPC_per_vm_per_server.sum(axis = 1)

        filter_zero = np.where(aux > 0)[0]

        if len(filter_zero) == 0:
            return 1
        else:
            return (num_p_cpus_per_server[filter_zero]/aux[filter_zero]).mean()

    def cpu_oc_hpc(self, x):
        # AVG OC CPU
        return (np.matmul(x, self.w[:,0]) / self.base_c[:,0]).mean()

    def ram_oc_hpc(self, x):
        return (np.matmul(x, self.w[:,1]) / self.base_c[:,1]).mean()

    def apply_tau_max_min(self):
        self.pheromone[self.pheromone < self.tau_min] = self.tau_min
        self.pheromone[self.pheromone > self.tau_max] = self.tau_max