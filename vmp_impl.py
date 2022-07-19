from vmp.aco_vmp import *
import numpy as np
from utils import *


def aco_for_vmp(p, c, w, oc, n_ants, max_iter, alpha, beta, rho, kp_first=True, plot = False, w1=1, w2=1, w3=1):
    evolution = np.zeros(max_iter)
    ants_evol = np.zeros(max_iter)

    acs = AcoVmp(p=p, c=c, w=w, oc=oc, n_ants=n_ants, max_iter=max_iter, alpha=alpha, beta=beta, rho=rho, kp_first=kp_first, w1=w1, w2=w2, w3=w3)

    acs.init_random_pheromone_trails()

    LONG_VAL_ITER = len(p) + 1

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
        #ants_evol[i] = np.mean([np.dot(ant.s, acs.p).sum() for ant in acs.ants])
    
    # + str(np.array(acs.best_fit) * 1.0) + ", "
    solution = acs.p[np.array(acs.best_fit).sum(axis=0) == 1]
    total_vms = len(solution)
    total_hpcs = len(solution[solution ==2])
    print("SOLUTION: "  + str(solution) + ", PROFIT: " + str(acs.best_fit_profit) + ", #HPC: " + str(total_hpcs) + ", BE: " + str(total_vms - total_hpcs))
    if plot:
        plot_mkp_solution3(acs, evolution)