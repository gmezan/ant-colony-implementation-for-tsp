from TSP.acs_tsp import AcsTsp
from TSP.Ant import *
from utils import *
import numpy as np
import pandas as pd

DEFAULT_DIST = 1000000
INSTANCE = 'resources/instance1.csv'
NUMBER_OF_ANTS = 10
N_ITERS = 200
ALPHA = 0.7
BETA = 1.2
RHO = 0.35


def acs_for_tsp(instance: str) -> None:
    print(">> Solving TSP with ACS")
    aco, graph, nodes = initialize_data(instance)

    aco.init_random_pheromone_trails()

    while(not aco.is_terminate()):
        aco.start_ants()

        while(not aco.are_ants_done()):
            aco.move_ants()

        aco.global_update_trails()

    print("DONE!")
    final_tour = [nodes[t] for t in aco.best_tour]
    print("Final tour: " + str(final_tour))
    print("Length: " + str(aco.best_tour_length))
    plot_solution(graph, final_tour)


# procedure InitializeData
def initialize_data(instance: str):
    graph, nodes = read_instance(instance) # ReadInstance
    
    # ComputeDistances
    N = len(nodes)
    d = np.ones((N,N)) * DEFAULT_DIST
    for key, value in graph.items():
        src_idx = nodes.index(key[0])
        dst_idx = nodes.index(key[1])

        d[src_idx][dst_idx] = value
        d[dst_idx][src_idx] = value

    # InitializeParameters
    aco = AcsTsp(d, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)

    return aco, graph, nodes


acs_for_tsp(INSTANCE)