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
def generate_vmp_instance(M, N, D = 2):
    # M items, N knapsacks

    aux = []

    c = []
    p = []
    w = []
    oc = []

    p_chooser = np.array([1,1,1,2,2])

    # CPUs
    w_d0_chooser = np.array([0.5,1,2,4,6,8,10,12,24])
    # RAM
    w_d1_chooser = np.array([0.5,1,2,4,6,8,10,12,24,32,64])
    # Other d
    w_dn_chooser = np.array([1,2,4,6,8,10,12,14,16,18,20])


    COEFF_SERVER = 4
    # CPUs
    c_d0_chooser = np.array([0.5,1,2,4,6,8,10,12,24]) * COEFF_SERVER
    # RAM
    c_d1_chooser = np.array([0.5,1,2,4,6,8,10,12,24,32,64])* COEFF_SERVER
    # Other d
    c_dn_chooser = np.array([1,2,4,6,8,10,12,14,16,18,20])* COEFF_SERVER

    for j in range(M):
        p.append(np.random.choice(p_chooser))
        aux = []
        for l in range(D):
            if l == 0:
                aux.append(np.random.choice(w_d0_chooser))
            elif l == 1:
                aux.append(np.random.choice(w_d1_chooser))
            else:
                aux.append(np.random.choice(w_dn_chooser))

        w.append(aux)

    for i in range(N):
        aux = []
        for l in range(D):
            if l == 0:
                aux.append(np.random.choice(c_d0_chooser))
            elif l == 1:
                aux.append(np.random.choice(c_d1_chooser))
            else:
                aux.append(np.random.choice(c_dn_chooser))
        c.append(aux)

    for i in range(N):
        aux = []
        for l in range(D):
            if l == 0:
                aux.append(16.0)
            elif l == 1:
                aux.append(5.0)
            else:
                aux.append(4.0)
        oc.append(aux)

    filename = str(uuid.uuid4()) + ".py"

    with open(VMP_PATH + filename, 'w') as f:
        f.write("c = " + str(c))
        f.write("\n")
        f.write("p = " + str(p))
        f.write("\n")
        f.write("w = " + str(w))
        f.write("\n")
        f.write("oc = " + str(oc))
        f.write("\n")

    print(filename)
    return filename

generate_vmp_instance(45,15,2)
