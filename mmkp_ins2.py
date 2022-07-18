from mmkp import *
import numpy as np
# http://elib.zib.de/pub/mp-testdata/ip/sac94-suite/text.en.html

def weish03():
    NUMBER_OF_ANTS = 5
    N_ITERS = 100
    ALPHA = 1.0
    BETA = 1.7
    RHO = 0.1
    KP_FIRST = True
    PLOT = True
    #p, c, w, s = read_instace_kp("p08")
    #print(np.matmul(p,s).sum())
    
    c = [[480,800,500,300,620]]

    p = [360, 83,59,130,431,67,230,52,93,125,670,892,600,38,48,147,78,256,63,17,120,164,432,35,92,110,22,42,50,323]

    w = [[7,0,30,22,80,94,11,81,70,64,59,18,0,36,3,8,15,42,9,0,42,47,52,32,26,48,55,6,29,84],
        [8,66,98,50,0,30,0,88,15,37,26,72,61,57,17,27,83,3,9,66,97,42,2,44,71,11,25,74,90,20],
        [3,74,88,50,55,19,0,6,30,62,17,81,25,46,67,28,36,8,1,52,19,37,27,62,39,84,16,14,21,5],
        [21,40,0,6,82,91,43,30,62,91,10,41,12,4,80,77,98,50,78,35,7,1,96,67,85,4,23,38,2,57],
        [94,86,80,92,31,17,65,51,46,66,44,3,26,0,39,20,11,6,55,70,11,75,82,35,47,99,5,14,23,38]
    ]
    w= np.transpose(np.array(w)).tolist()
    #
    acs_for_mmkp(p=p, c=c, w=w, n_ants=NUMBER_OF_ANTS, max_iter=N_ITERS, alpha=ALPHA, beta=BETA, rho=RHO, kp_first=KP_FIRST, plot=PLOT)

weish03()