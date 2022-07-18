import numpy as np
import uuid

P = 100
W = 20
C = 36

VMP_PATH = "resources/simul/vmp/"


"""
    # 5 items
    p = [3,2,3,4,5,4]
    # 4 servers with 3 constraints each
    c = [[7,9,8],[6,9,8],[6,5,8],[9,5,6]]
    # items weights 5x3
    w = [[1,2,3],[3,2,1],[1,2,1],[2,2,1],[3,1,1],[1.5,2,1]]
"""
def generate_vmp_instance(M, N, D):
    # M items, N knapsacks

    aux = []

    c = []
    p = []
    w = []


    for j in range(M):
        p.append(int(np.random.rand() * P))
        aux = []
        for l in range(D):
            aux.append(np.random.rand() * W)
        w.append(aux)

    for i in range(N):
        aux = []
        for l in range(D):
            aux.append(np.random.rand() * C)
        c.append(aux)

    filename = str(uuid.uuid4()) + ".py"

    with open(VMP_PATH + filename, 'w') as f:
        f.write("c = " + str(c))
        f.write("\n")
        f.write("p = " + str(p))
        f.write("\n")
        f.write("w = " + str(w))
        f.write("\n")

    print(filename)
    return filename

generate_vmp_instance(30,12,3)
