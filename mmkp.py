from mkp.aco_mkp import *
import numpy as np
from utils import *

LONG_VAL_ITER = 1000

def demo():

    NUMBER_OF_ANTS = 20
    N_ITERS = 100
    ALPHA = 1
    BETA = 2
    RHO = 0.35

    """
    i -> 4
    j -> 5
    l -> 3
    """

    # 5 items
    p = [3,2,3,4,5,4]
    # 4 servers with 3 constraints each
    c = [[7,9,8],[6,9,8],[6,5,8],[9,5,6]]
    # items weights 5x3
    w = [[1,2,3],[3,2,1],[1,2,1],[2,2,1],[3,1,1],[1.5,2,1]]

    aco = AcoMkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO)

    # mocked solution x[i][j] 4x5
    x = np.zeros((4,6))
    x[3][0] = 1
    x[1][1] = 1
    x[0][2] = 1
    x[3][4] = 1
    x[3][3] = 0

    ans = aco.compute_pseudo_utility_eta(x)
    #print(w)
    #aco.ants[0].s = x
    #print(aco.calculate_allowed(0))

    aco.start_ants()

    aco.add_item(0)
    print(aco.ants[0].s * 1)
    #print(aco.compute_avg_tightness_delta(aco.ants[0].s).mean(axis=0))
    aco.add_item(0)
    
    print(aco.ants[0].s * 1)

def acs_for_mmkp(p, c, w, n_ants, max_iter, alpha, beta, rho, kp_first=True, plot = False):
    evolution = np.zeros(max_iter)
    ants_evol = np.zeros(max_iter)

    acs = AcoMkp(p=p, c=c, w=w, n_ants=n_ants, max_iter=max_iter, alpha=alpha, beta=beta, rho=rho, kp_first=kp_first)

    acs.init_random_pheromone_trails()

    LONG_VAL_ITER = len(p)

    for i in range(max_iter):
        acs.start_ants()
        #print("iteration: " + str(acs.counter))
        for k in range(len(acs.ants)):
            count = 0
            while np.any(acs.ants[k].allowed):
                acs.add_item(k)
                count += 1
                if count > LONG_VAL_ITER:
                    acs.ants[k].allowed.fill(0)
                #print("hi: " + str(k))

            # Calculate L_k and saves the best solution so far
            acs.update_objective_func(k)

        acs.update_trails()
        evolution[i] = acs.best_fit_profit
        ants_evol[i] = np.mean([np.dot(ant.s, acs.p).sum() for ant in acs.ants])
    print("SOLUTION: \n" + str(np.array(acs.best_fit) * 1.0) + ", " + str(acs.p[np.array(acs.best_fit).sum(axis=0) == 1]) + ", PROFIT: " + str(acs.best_fit_profit))
    if plot:
        plot_mkp_solution2(acs, evolution, ants_evol)

