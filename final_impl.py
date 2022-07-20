import numpy as np
from vmp_impl import round_robin_vmp
import matplotlib.pyplot as plt

NUMBER_OF_ANTS = 8
N_ITERS = 75
ALPHA = 1.0
BETA = 1.2
RHO = 0.03
KP_FIRST = False
PLOT = True
W1 = 1
W2 = 1
W3 = 2.0


# 45 VMs, 15 PMs, 2 D
def ins01():
    c = [[40.0, 24.0], [40.0, 48.0], [48.0, 16.0], [2.0, 32.0], [2.0, 256.0], [96.0, 48.0], [32.0, 96.0], [2.0, 48.0], [40.0, 4.0], [40.0, 2.0], [40.0, 96.0], [96.0, 4.0], [40.0, 2.0], [32.0, 96.0], [4.0, 48.0]]
    p = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2]
    w = [[12.0, 0.5], [2.0, 4.0], [2.0, 10.0], [1.0, 4.0], [2.0, 12.0], [10.0, 12.0], [1.0, 0.5], [1.0, 64.0], [8.0, 6.0], [12.0, 6.0], [6.0, 6.0], [12.0, 2.0], [4.0, 1.0], [8.0, 6.0], [24.0, 64.0], [12.0, 10.0], [1.0, 64.0], [8.0, 6.0], [4.0, 8.0], [10.0, 4.0], [24.0, 4.0], [8.0, 6.0], [8.0, 64.0], [12.0, 0.5], [4.0, 64.0], [6.0, 10.0], [6.0, 1.0], [4.0, 24.0], [12.0, 64.0], [10.0, 32.0], [24.0, 8.0], [12.0, 6.0], [12.0, 64.0], [24.0, 32.0], [8.0, 24.0], [1.0, 0.5], [2.0, 10.0], [12.0, 2.0], [1.0, 32.0], [4.0, 12.0], [4.0, 24.0], [8.0, 64.0], [10.0, 10.0], [0.5, 0.5], [1.0, 8.0]]
    oc = [[16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0]]

    return c,p,w,oc



def run_algorithms():

    c,p,w,oc = ins01()

    # aco-vmp
    #aco, evol_aco, sol_aco = aco_for_vmp_headless_2(p=p, c=c, w=w, oc=oc, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST)

    # greedy
    #greedy, evol_greedy, sol_greedy = aco_for_vmp_headless_2(p=p, c=c, w=w, oc=oc, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=0, beta=BETA, rho=RHO, kp_first=KP_FIRST)

    # rr
    rr, evol_rr, sol_rr = round_robin_vmp(p=p, c=c, w=w, oc=oc, max_iter=N_ITERS)

    #plt.plot(evol_aco, color='b')
    #plt.plot(evol_greedy, color='r')
    plt.plot(evol_rr, color='g')
    #plt.legend(["ACO-VMP", "Greedy", "Round Robin"])
    plt.show()

run_algorithms()
