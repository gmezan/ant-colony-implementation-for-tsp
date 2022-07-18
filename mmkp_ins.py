from mmkp import *
import numpy as np

BASE_PATH = "instances-fsu-edu/mkp/"
BASE_PATH_KP = "instances-fsu-edu/kp/"

def ins1():
    NUMBER_OF_ANTS = 20
    N_ITERS = 100
    ALPHA = 1
    BETA = 2
    RHO = 0.35
    KP_FIRST = True
    # 5 items
    p = [3,2,3,4,5,4,5,2]
    # 4 servers with 3 constraints each
    c = [[4,5,5],[3,5,5],[3,2,5],[4,2,3]]
    # items weights 5x3
    w = [[1,2,3],[3,2,1],[1,2,1],[2,2,1],[3,1,1],[1.5,2,1],[3,2,3],[1,2,1]]

    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST)


def p01():
    NUMBER_OF_ANTS = 32
    N_ITERS = 150
    ALPHA = 1
    BETA = 3
    RHO = 0.5
    KP_FIRST = True
    p, c, w, s = read_instace_mkp("p01")
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST)

    print(np.matmul(p,s).sum())

def p05():
    NUMBER_OF_ANTS = 32
    N_ITERS = 150
    ALPHA = 1
    BETA = 3
    RHO = 0.5
    KP_FIRST = True
    p, c, w, s = read_instace_mkp("p05")
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST)

    print(np.matmul(p,s).sum())

def p06():
    NUMBER_OF_ANTS = 32
    N_ITERS = 150
    ALPHA = 1
    BETA = 3
    RHO = 0.5
    KP_FIRST = True
    p, c, w, s = read_instace_mkp("p06")
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST)

    print(np.matmul(p,s).sum())
    
def kp_p06():
    NUMBER_OF_ANTS = 20
    N_ITERS = 100
    ALPHA = 1.0
    BETA = 1.5
    RHO = 0.05
    KP_FIRST = True
    p, c, w, s = read_instace_kp("p06")
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST)

    print(np.matmul(p,s).sum())

def kp_p07():
    NUMBER_OF_ANTS = 20
    N_ITERS = 100
    ALPHA = 1.0
    BETA = 1.5
    RHO = 0.05
    KP_FIRST = True
    p, c, w, s = read_instace_kp("p07")
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST)
    print(np.matmul(p,s).sum())

def kp_p08():
    NUMBER_OF_ANTS = 32
    N_ITERS = 200
    ALPHA = 1.0
    BETA = 1.7
    RHO = 0.1
    KP_FIRST = False
    PLOT = True
    p, c, w, s = read_instace_kp("p08")
    print(np.matmul(p,s).sum())
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST, plot=PLOT)
    


def read_instace_mkp(n):
    p = []
    c = []
    w = []
    s = []
    with open(BASE_PATH + n + "_c.txt") as f:
        lines = f.readlines()
        for i in lines:
            for j in i.strip().split('  '):
                c.append([int(j)])
    
    with open(BASE_PATH + n + "_w.txt") as f:
        lines = f.read().splitlines()
        for i in lines:
            w.append([int(i)])

    with open(BASE_PATH + n + "_p.txt") as f:
        lines = f.read().splitlines()
        for i in lines:
            p.append(int(i))

    with open(BASE_PATH + n + "_s.txt") as f:
        lines = f.read().splitlines()
        for i in lines:
            aa = i.split(' ')
            s.append([int(aa[0]), int(aa[1])])

    return p, c, w, s

def read_instace_kp(n):
    p = []
    c = []
    w = []
    s = []
    with open(BASE_PATH_KP + n + "_c.txt") as f:
        lines = f.read().splitlines()
        for i in lines:
            c.append([int(i)])
    
    with open(BASE_PATH_KP + n + "_w.txt") as f:
        lines = f.read().splitlines()
        for i in lines:
            w.append([int(i)])

    with open(BASE_PATH_KP + n + "_p.txt") as f:
        lines = f.read().splitlines()
        for i in lines:
            p.append(int(i))

    with open(BASE_PATH_KP + n + "_s.txt") as f:
        lines = f.read().splitlines()
        for i in lines:
            s.append([int(i)])

    return p, c, w, s


kp_p08()