import numpy as np

class Ant:
    """
    n: number of items
    m: number of knapsacks
    """
    def __init__(self, m, n) -> None:
        # value of objective function
        self.profit = 0.0
        # list of items selected in a time t / knapsack: true if taken
        self.s = np.zeros((m,n), dtype=bool)
        # integer/boolean visited[n] % visited items: true if visited
        self.tabu_list = np.zeros((m,n), dtype=bool)

        # termination flag
        self.done = False

        # items allowed for ant: true if allowed
        self.allowed = np.ones((m,n), dtype=bool)


    def reset(self):
        self.done = False
        self.step = 0
        self.profit = 0.0
        self.tabu_list.fill(False)
        self.s.fill(False)
        self.allowed.fill(True)

    def add(self, i: int, j:int):
        self.s[i][j] = True
        self.tabu_list[i][j] = True