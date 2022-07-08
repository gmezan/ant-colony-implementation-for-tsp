from Ant import Ant
import numpy as np

class AntColonyOptimizationTSP:
    
    def __init__(self, dist, nn_list, pheromone, choice_info, n_ants, max_iter = 1000000, alpha = 0.5, beta = 0.5, rho = 0.5) -> None:
        # integer dist[n][n] % distance matrix
        self.dist = dist;
        # integer nn_list[n][nn] % matrix with nearest neighbor lists of depth nn
        self.nn_list = nn_list
        # real pheromone[n][n] % pheromone matrix (pheromone trail tau)
        self.pheromone = pheromone;
        # real choice_info[n][n] % combined pheromone and heuristic information
        self.choice_info = choice_info;

        # single_ant ant[m] % structure of type single_ant
        self.ants = [Ant(len(dist)) for _ in range(n_ants) ]

        # Iteration parameters
        self.terminate = False;
        self.counter = 0
        self.max_iter = max_iter;

        # Hyper parameters:
        self.alpha = alpha  # weight for trails
        self.beta = beta    # weight for lengths (weights)
        self.rho = rho      # trail/pheromone evaporation [0, 1]

        # Parameters
        self.source = None
        self.target = None
        self.best_tour = []

        # Private attr
        self.__M = len(self.ants)
        self.__N = len(self.dist)

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

    def construct_solutions(self):
        # empty ant
        for k in range(self.__M):
            self.ants[k].visited.fill(False)

        step = 0

        chooser = np.array(range(self.__N)) # chooser = [0, 1, ..., N -1 ]

        # choose initial random node for each ant
        for k in range(self.__M):
            r = np.random.choice(chooser)
            self.ants[k].tour[step] = r
            self.ants[k].visited[r] = True

        # each ant construct a complete tour
        while (step < self.__N):
            step += 1
            for k in range(self.__M):
                self.as_decision_rule(k, step)
            
        # ants move back to the initial city
        for k in range(self.__M):
            self.ants[k].tour[self.N] = self.ants[k].tour[0]
            self.ants[k].tour_length = self.compute_tour_length(k)

    # k: the ant identifier, step: counter for construction step
    def as_decision_rule(self, k, step):
        c = self.ants[k].tour[step - 1]
        sum_probabilities = 0.0;

        for j in range(self.__N):
            if self.ants[k].visited[j]:
                pass


        

    def compute_ant_decision(self):
        # a^i_j(t) = 
        return None
