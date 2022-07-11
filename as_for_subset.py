from Subset.acs_subset import AsSubset
import numpy as np
from utils import plot_subset_solution, plot_subset_solution2

DEFAULT_DIST = 1000000
INSTANCE = 'resources/instance1.csv'
NUMBER_OF_ANTS = 50
N_ITERS = 100
ALPHA = 1
BETA = 3
RHO = 0.5

# sol = [0,1,1,1] ?
P = [4,10,2,6]
C = [8,12,10]
R = [[4,4,2,1], [6,6,3,3], [4,4,2,2]]

def subset_basic_example():
    p = [4,10,2,6]
    c = [8,12,10]
    r = [[4,4,2,1], [6,6,3,3], [4,4,2,2]]

    acs = AsSubset(p, c, r, NUMBER_OF_ANTS)
    print(acs.compute_pseudo_utility_eta([0,0,0,1]))


def acs_for_subset(p, c, r, n_ants, max_iter, alpha, beta, rho):
    evolution = []
    ants_evol = []

    acs = initialize(p=p, c=c, r=r, n_ants=n_ants, max_iter=max_iter, alpha=alpha, beta=beta, rho=rho)

    acs.init_random_pheromone_trails()

    while not acs.is_terminate():
        N_max = 0
        acs.start_ants()
        #print("iteration: " + str(acs.counter))
        for k in range(len(acs.ants)):
            N_items = 0
            
            while np.any(acs.ants[k].allowed):
                added = acs.add_item(k)
                if added:
                    #print("adding item for ant: " + str(k))
                    N_items += 1

            # Calculate L_k and saves the best solution so far
            acs.update_objective_func(k)
            N_max = np.max((N_max, N_items))

        acs.update_trails()
        evolution.append(acs.best_fit_profit)
        ants_evol.append(np.mean([np.dot(acs.p, ant.s) for ant in acs.ants]))
    print("SOLUTION: " + str(acs.best_fit + 0) + ", " + str(acs.p[acs.best_fit]) + ", PROFIT: " + str(acs.best_fit_profit))
    plot_subset_solution2(acs, evolution, ants_evol)


def initialize(p, c, r, n_ants, max_iter, alpha, beta, rho) -> AsSubset:
    acs = AsSubset(p=p, c=c, r=r, n_ants=n_ants, max_iter=max_iter, alpha=alpha, beta=beta, rho=rho)
    return acs

def run_instance_0():
    NUMBER_OF_ANTS = 8
    N_ITERS = 25
    ALPHA = 1.0
    BETA = 1.5
    RHO = 0.05
    # sol = [0,1,1,1] ?
    P = [4,10,2,6]
    C = [8,12,10]
    R = [[4,4,2,1], [6,6,3,3], [4,4,2,2]]
    acs_for_subset(p=P, c=C, r=R, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)

def run_instance_1():
    NUMBER_OF_ANTS = 12
    N_ITERS = 75
    ALPHA = 1.0
    BETA = 1.5
    RHO = 0.1
    # sol = SOLUTION: [0 1 0 1 0 1 1 1], [10  6  4  8 12], PROFIT: 40 ????
    P = [4,10,2,6,2,4,8,12]
    C = [18,22,20]
    R = [[4,4,2,1,2,4,3,5], [6,6,3,3,3,2,4,4], [4,4,2,2,1,3,5,4]]
    acs_for_subset(p=P, c=C, r=R, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)

def run_instance_2():
    NUMBER_OF_ANTS = 20
    N_ITERS = 100
    ALPHA = 1.0
    BETA = 1.5
    RHO = 0.05
    # sol = SOLUTION: [0 1 0 1 0 1 1 1], [10  6  4  8 12], PROFIT: 40 ????
    P = [4,10,2,6,2,4,8,12]
    C = [18,22,20,30]
    R = [[4,4,2,1,2,4,3,5], [6,6,3,3,3,2,4,4], [4,4,2,2,1,3,5,4], [5,5,1,2,5,8,2,3]]
    acs_for_subset(p=P, c=C, r=R, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)


run_instance_2()
