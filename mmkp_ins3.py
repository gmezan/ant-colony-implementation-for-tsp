# http://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html
import numpy as np
from mmkp import *

def mknap1_1():
    NUMBER_OF_ANTS = 5
    N_ITERS = 100
    ALPHA = 0.0
    BETA = 1.7
    RHO = 0.1
    KP_FIRST = True
    PLOT = True
    #p, c, w, s = read_instace_kp("p08")
    #print(np.matmul(p,s).sum())
    

    c = [[80, 96, 20, 36, 44, 48, 10, 18, 22, 24]]

    p = [100, 600, 1200, 2400, 500, 2000]

    
    w = [[8,12,13,64,22,41],[8,12,13,75,22,41]
,[3,6,4,18,6,4]
,[5,10,8,32,6,12]
,[5,13,8,42,6,20]
,[5,13,8,48,6,20]
,[0,0,0,0,8,0]
,[3,0,4,0,8,0]
,[3,2,4,0,8,4]
,[3,2,4,8,8,4]
    ]
    w= np.transpose(np.array(w)).tolist()
    #
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST, plot=PLOT)

mknap1_1()