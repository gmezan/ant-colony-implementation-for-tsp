import numpy as np

class Ant:
    def __init__(self, n) -> None:
        # value of objective function
        self.profit = 0.0
        self.step = 0
        # list of items selected in a time t / knapsack: true if taken
        self.s = np.zeros((n,), dtype=bool)
        # integer/boolean visited[n] % visited items: true if visited
        self.tabu_list = np.zeros((n,), dtype=bool)

        # termination flag
        self.done = False

        # items allowed for ant: true if allowed
        self.allowed = np.ones((n,), dtype=bool)


    def reset(self):
        self.done = False
        self.step = 0
        self.profit = 0.0
        self.tabu_list.fill(False)
        self.s.fill(False)
        self.allowed.fill(True)

    def add(self, x: int):
        self.step += 1
        self.s[x] = True
        self.tabu_list[x] = True