from TSP.AntColonyOptimizationTSP import AntColonyOptimizationTSP as ACO
from TSP.Ant import Ant
from utils import *
import numpy as np
import pandas as pd

DEFAULT_DIST = 10000
INSTANCE = 'resources/instance0.csv'
NUMBER_OF_ANTS = 10

""" procedure ACOforTSP
        InitializeData
        while (not terminate) do
            ConstructSolutions
            LocalSearch
            UpdateStatistics
            UpdatePheromoneTrails
        end-while
    end-procedure """
def aco_for_tsp(instance: str) -> None:
    # ---- InitializeData ----
    aco = initialize_data(instance, 0, 3)

    # ---- while (not terminate) do ----
    while(not aco.is_terminate()):

        # ---- ConstructSolutions ----
        aco.construct_solutions()
        # ---- LocalSearch ----

        # ---- UpdateStatistics ----

        # ---- UpdatePheromoneTrails ----


# procedure InitializeData
def initialize_data(instance: str, source_node, target_node) -> ACO:

    # ReadInstance
    graph, nodes = read_instance(instance)
    plot_graph(graph)
    
    # ComputeDistances
    N = len(nodes)
    d = np.ones((N,N)) * DEFAULT_DIST
    for key, value in graph.items():
        src = key[0]
        dst = key[1]

        src_idx = nodes.index(src)
        dst_idx = nodes.index(dst)

        d[src_idx][dst_idx] = value
        d[dst_idx][src_idx] = value

    print("\ninitialize_data: ComputeDistances")
    print(d)


    # ComputeNearestNeighborLists
    l_nn_list = []
    for row in d:
        df_d = pd.DataFrame({'d_i': list(row), 'i': range(len(row))}).sort_values(by='d_i', ascending=True)
        l_nn_list.append(df_d['i'].to_list())

    nn_list = np.array(l_nn_list)
    print("\ninitialize_data: ComputeNearestNeighborLists")
    print(nn_list)

    # ComputeChoiceInformation
    pheromone = np.zeros((N,N))
    choice_info = np.zeros((N,N))

    # InitializeAnts
    M = NUMBER_OF_ANTS

    # InitializeParameters
    alpha = 0.5
    beta = 0.5
    rho = 0.5
    max_iter = 100000

    aco = ACO(d, nn_list, pheromone, choice_info, M, max_iter, alpha, beta, rho)
    aco.source = source_node
    aco.target = target_node

    # InitializeStatistics

    # TODO: compute time, mem, etc

    return aco


aco_for_tsp(INSTANCE)