from TSP.acs_tsp import AcsTsp
from TSP.Ant import *
from utils import *
import numpy as np
import pandas as pd

DEFAULT_DIST = 1000000
INSTANCE = 'resources/instance1.csv'
NUMBER_OF_ANTS = 20
N_ITERS = 100
ALPHA = 1.0
BETA = 1.2
RHO = 0.03

evolution = []
ants_evol = []

# Solves: Fin Hamiltonian cycle of G = (V,E) for TSP
# https://mathworld.wolfram.com/HamiltonianCycle.html
def acs_for_tsp(instance: str) -> None:
    print(">> Solving TSP with ACS")
    aco, graph, nodes = initialize_data(instance)

    aco.init_random_pheromone_trails()

    while(not aco.is_terminate()):
        aco.start_ants()

        while(not aco.are_ants_done()):
            aco.move_ants()

        ants_evol.append(np.mean([ant.tour_length for ant in aco.ants if ant.tour_length < DEFAULT_DIST]))

        aco.global_update_trails()
        evolution.append(aco.best_tour_length)

    final_tour = [nodes[t] for t in aco.best_tour]
    print("\nFinal tour: " + str(final_tour))
    print("Length: " + str(aco.best_tour_length))
    #plot_solution(graph=graph, final_tour=final_tour)
    plot_results(graph=graph, final_tour=final_tour, evolution=evolution, ants_evol=ants_evol)


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