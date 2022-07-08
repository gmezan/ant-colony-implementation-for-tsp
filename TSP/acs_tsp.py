from .Ant import *
import numpy as np

TINY_VAL = 0.0001
LONG_VAL = 10000

class AcsTsp:
    # first implementing Ant Colony System For TSP

    def __init__(self, dist, n_ants, max_iter = 1000000, alpha = 0.5, beta = 1, rho = 0.3) -> None:
        # Private attr
        self.__M = n_ants
        self.__N = len(dist)
        self.__chooser = np.array(range(self.__N)) # chooser = [0, 1, ..., N -1 ]
        
        # integer dist[n][n] % distance matrix
        self.dist = dist;
        # real pheromone[n][n] % pheromone matrix (pheromone trail tau)
        self.pheromone = None;
        # single_ant ant[m] % structure of type single_ant
        self.ants = [Ant(len(dist)) for _ in range(n_ants) ]

        # Hyper parameters:
        self.alpha = alpha  # weight for trails
        self.beta = beta    # weight for lengths (weights) (recommended beta > alpha)
        self.rho = rho      # trail/pheromone evaporation [0, 1]
        self.q_o = np.random.rand()
        self.max_iter = max_iter;
        self.tau_o = 0.005

        # Parameters
        self.terminate = False;
        self.counter = 0
        self.best_tour = None
        self.best_tour_length = None


    def init_random_pheromone_trails(self):
        self.pheromone = np.random.rand(self.__N, self.__N)


    def start_ants(self):
        for k in range(self.__M):
            self.ants[k].reset()
            self.ants[k].visit(np.random.choice(self.__chooser), self.dist)

    
    def move_ants(self):
        # computing heuristic for each ant
        for k in range(self.__M):
            j = None # j is next node
            i = self.ants[k].get_current_node()
            
            aux_1 = self.pheromone[i] ** self.alpha # tau_i ^ alpha
            aux_2 = (1/self.dist[i]) ** self.beta   # eta_i ^ beta

            # computing ant decision table / discard visited nodes
            a_j = (aux_1 * aux_2 / np.dot(aux_1, aux_2)) * (self.ants[k].visited == False)

            # if already visited every city
            if np.all(a_j == 0):
                if not self.ants[k].rounded:
                    self.ants[k].rounded = True
                    j = self.ants[k].tour[0]
                    # last ant move
                    self.ants[k].visit(j, self.dist)
                else:
                    self.ants[k].done = True
            else:
                # computing probabilities
                probs = a_j / np.sum(a_j)

                # applying meta heuristic with q_o 
                rand = np.random.rand()
                aux = ( probs > rand) * probs
                if (rand <= self.q_o) or np.all(aux == 0):
                    j = np.argmax(probs)
                else:
                    aux[aux <= 0.0] = LONG_VAL
                    j = np.argmin(aux)

                # move the ant using meta heuristic
                self.ants[k].visit(j, self.dist)


        # update trail followed by the ants
        delta_tau = np.zeros((self.__N, self.__N), dtype='float64')

        for k in range(self.__M):
            if not self.ants[k].done:
                i = self.ants[k].get_previous_node()
                j = self.ants[k].get_current_node()
                L = TINY_VAL if self.ants[k].tour_length <= 0 else self.ants[k].tour_length
                delta_tau[i][j] += 1 / L
                delta_tau[j][i] += 1 / L # fully conected graph 

        self.pheromone = self.pheromone * (1 - self.rho) + self.rho * delta_tau


    def global_update_trails(self):
        if self.best_tour == None:
            self.best_tour = self.ants[0].tour
            self.best_tour_length = self.ants[0].tour_length
        
        # find best tour so far
        for ant in self.ants:
            if (ant.tour_length > 0 and (ant.tour_length < self.best_tour_length)):
                self.best_tour = ant.tour
                self.best_tour_length = ant.tour_length

        # global update
        aux = list(self.best_tour)
        aux.append(self.best_tour[0])
        delta_tau = np.zeros((self.__N, self.__N), dtype='float64')

        if (self.best_tour_length == 0):
            return

        for l in range(len(self.best_tour)):
            L = 1 / self.best_tour_length
            delta_tau[aux[l]][aux[l + 1]] += L
            delta_tau[aux[l + 1]][aux[l]] += L

        self.pheromone = self.pheromone * (1 - self.rho) + self.rho * delta_tau


    """
    The program stops if at least one termination condition applies. Possible termination
    conditions are: (1) the algorithm has found a solution within a predefined distance
    from a lower bound on the optimal solution quality; (2) a maximum number of tour
    constructions or a maximum number of algorithm iterations has been reached; (3) a
    maximum CPU time has been spent; or (4) the algorithm shows stagnation behavior.
    """
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