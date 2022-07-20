import numpy as np
from vmp_impl import *
import matplotlib.pyplot as plt

NUMBER_OF_ANTS = 6
N_ITERS = 40
ALPHA = 1.0
BETA = 2.5
RHO = 0.03
KP_FIRST = False
PLOT = True
W1 = 1
W2 = 1
W3 = 1.5

def get_ins01():
    c = [[32.0, 128.0], [4.0, 24.0], [8.0, 4.0], [48.0, 24.0], [16.0, 256.0], [32.0, 256.0]]
    p = [1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]
    w = [[6.0, 12.0], [2.0, 12.0], [10.0, 12.0], [6.0, 24.0], [6.0, 1.0], [6.0, 24.0], [12.0, 2.0], [12.0, 1.0], [4.0, 8.0], [12.0, 4.0], [6.0, 8.0], [2.0, 24.0], [8.0, 24.0], [12.0, 6.0], [12.0, 32.0], [24.0, 24.0], [1.0, 1.0], [6.0, 8.0], [12.0, 32.0], [12.0, 6.0], [12.0, 32.0], [2.0, 12.0], [10.0, 6.0], [8.0, 4.0], [8.0, 10.0]]
    oc = [[16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0], [16.0, 5.0]]

    return c, p, w, oc


def rho_tunning():

    rhos = 1 * np.array(range(100)) / 100

    c,p,w,oc = get_ins01()

    solutions = []

    for rho in rhos:
        solutions.append(aco_for_vmp_headless(p=p, c=c, w=w, oc=oc, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=rho, kp_first=KP_FIRST))

    plt.plot(rhos, np.array(solutions))
    plt.xlabel("rho")
    plt.ylabel("Obj Function")
    plt.show()

def beta_tunning():

    betas = 5 * np.array(range(100)) / 100

    c,p,w,oc = get_ins01()

    solutions = []

    for beta in betas:
        solutions.append(aco_for_vmp_headless(p=p, c=c, w=w, oc=oc, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=beta, rho=RHO, kp_first=KP_FIRST))

    plt.plot(betas, np.array(solutions))
    plt.xlabel("beta")
    plt.ylabel("Obj Function")
    plt.show()

def alpha_tunning():

    alphas = 2.5 * np.array(range(100)) / 100

    c,p,w,oc = get_ins01()

    solutions = []

    for alpha in alphas:
        solutions.append(aco_for_vmp_headless(p=p, c=c, w=w, oc=oc, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=alpha, beta=BETA, rho=RHO, kp_first=KP_FIRST))

    plt.plot(alphas, np.array(solutions))
    plt.xlabel("alpha")
    plt.ylabel("Obj Function")
    plt.show()


def w3_tunning():

    w3s = 2.5 * np.array(range(100)) / 100

    c,p,w,oc = get_ins01()

    solutions = []

    for w3 in w3s:
        solutions.append(aco_for_vmp_headless(p=p, c=c, w=w, oc=oc, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST, w1=W1, w2=W2, w3=w3))

    plt.plot(w3s, np.array(solutions))
    plt.xlabel("alpha")
    plt.ylabel("Obj Function")
    plt.show()

w3_tunning()