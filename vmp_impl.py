from mmkp_ins5 import N_ITERS
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


def aco_for_vmp_headless(p, c, w, oc, n_ants, max_iter, alpha, beta, rho, kp_first=True, w1=1, w2=1, w3=1):

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
    
    return acs.best_fit_profit


def aco_for_vmp_headless_2(p, c, w, oc, n_ants, max_iter, alpha, beta, rho, kp_first=True, w1=1, w2=1, w3=1):
    evolution = np.zeros(max_iter)

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
    #total_vms = len(solution)
    #total_hpcs = len(solution[solution ==2])
    #print("SOLUTION: "  + str(solution) + ", PROFIT: " + str(acs.best_fit_profit) + ", #HPC: " + str(total_hpcs) + ", BE: " + str(total_vms - total_hpcs))
    return acs, evolution, solution

def round_robin_vmp(p, c, w, oc, max_iter):
    evolution = []

    acs = AcoVmp(p=p, c=c, w=w, oc=oc, n_ants=1)

    k = 0
    i_counter = 0
    i_aux = 0
    i_max = len(c) - 1

    acs.ants[k].reset()

    for j in range(len(p)):
        allowed = acs.calculate_allowed(k)
        j_placed = False
        i_aux = i_counter
        i_aux_counter = 0
        while (not j_placed) or (i_aux_counter >= i_max):
            # if placement is allowed
            if allowed[i_aux][j]:
                acs.ants[k].add(i_aux,j)
                # G(L_k) = Q * L_k
                acs.ants[k].profit = acs.compute_obj(acs.ants[k].s)
                j_placed = True
            else:
                i_aux_counter += 1
                if i_aux >= i_max:
                    i_aux = 0
                else:
                    i_aux += 1

            evolution.append(acs.ants[k].profit)
        
        if i_counter >= i_max:
            i_counter = 0
        else:
            i_counter += 1

    solution = acs.p[np.array(acs.ants[k].s).sum(axis=0) == 1]
    #total_vms = len(solution)
    #total_hpcs = len(solution[solution ==2])
    evolution = np.array(evolution)
    #print("SOLUTION: "  + str(solution) + ", PROFIT: " + str(acs.ants[k].profit) + ", #HPC: " + str(total_hpcs) + ", BE: " + str(total_vms - total_hpcs))
    if len(evolution) < max_iter:
        aux_evol = np.ones(max_iter)*acs.ants[k].profit
        aux_evol[0:len(evolution)] = evolution
    
    return acs, evolution, solution