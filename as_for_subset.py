from subset.acs_subset import AsSubset
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
    evolution = np.zeros(max_iter)
    ants_evol = np.zeros(max_iter)

    acs = initialize(p=p, c=c, r=r, n_ants=n_ants, max_iter=max_iter, alpha=alpha, beta=beta, rho=rho)

    acs.init_random_pheromone_trails()

    for i in range(max_iter):
        acs.start_ants()
        #print("iteration: " + str(acs.counter))
        for k in range(len(acs.ants)):
            while np.any(acs.ants[k].allowed):
                acs.add_item(k)

            # Calculate L_k and saves the best solution so far
            acs.update_objective_func(k)

        acs.update_trails()
        evolution[i] = acs.best_fit_profit
        ants_evol[i] = np.mean([np.dot(acs.p, ant.s) for ant in acs.ants])
    print("SOLUTION: " + str(np.array(acs.best_fit) + 0) + ", " + str(acs.p[acs.best_fit]) + ", PROFIT: " + str(acs.best_fit_profit))
    plot_subset_solution2(acs, evolution, ants_evol)


def initialize(p, c, r, n_ants, max_iter, alpha, beta, rho) -> AsSubset:
    acs = AsSubset(p=p, c=c, r=r, n_ants=n_ants, max_iter=max_iter, alpha=alpha, beta=beta, rho=rho)
    return acs

def run_instance_0():
    NUMBER_OF_ANTS = 4
    N_ITERS = 32
    ALPHA = 1.0
    BETA = 1.5
    RHO = 0.75
    # sol = [0,1,1,1] ?
    P = [4,10,2,6]
    C = [8,12,10]
    R = [[4,4,2,1], [6,6,3,3], [4,4,2,2]]
    acs_for_subset(p=P, c=C, r=R, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)

def run_instance_1():
    NUMBER_OF_ANTS = 8
    N_ITERS = 50
    ALPHA = 1.0
    BETA = 2
    RHO = 0.3
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


# Getting instances from: https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html

"""
442,525,511,593,546,564,617
"""
def run_instance_p06():
    NUMBER_OF_ANTS = 20
    N_ITERS = 100
    ALPHA = 1.0
    BETA = 1.5
    RHO = 0.05
    # sol = SOLUTION: [0 1 0 1 0 0 1], [525 593 617], PROFIT: 1735
    P = [442,525,511,593,546,564,617]
    C = [170]
    R = [[41,50,49,59,55,57,60]]
    acs_for_subset(p=P, c=C, r=R, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)

""" P07
"""
def run_instance_p07():
    NUMBER_OF_ANTS = 10
    N_ITERS = 100
    ALPHA = 1
    BETA = 2
    RHO = 0.35
    # sol = SOLUTION: [1 0 1 0 1 0 1 1 1 0 0 0 0 1 1], [135 149 156 173 184 192 229 240], PROFIT: 1458
    P = [135,139,149,150,156,163,173,184,192,201,210,214,221,229,240]
    C = [750]
    R = [[70,73,77,80,82,87,90,94,98,106,110,113,115,118,120]]
    acs_for_subset(p=P, c=C, r=R, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)


""" P08
382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684
"""
def run_instance_p08():
    NUMBER_OF_ANTS = 20
    N_ITERS = 200
    ALPHA = 1
    BETA = 2
    RHO = 0.4
    # sol = SOLUTION: [0 1 0 1 0 0 1], [525 593 617], PROFIT: 13549094
    P = [825594,1677009,1676628,1523970,943972,97426,69666,1296457,1679693,1902996,1844992,1049289,1252836,1319836,953277,2067538,675367,853655,1826027,65731,901489,577243,466257,369261]
    C = [6404180]
    R = [[382745,799601,909247,729069,467902,44328,34610,698150,823460,903959,853665,551830,610856,670702,488960,951111,323046,446298,931161,31385,496951,264724,224916,169684]]
    acs_for_subset(p=P, c=C, r=R, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)

run_instance_p08()
