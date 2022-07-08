import numpy as np

# % Representation of ants
class Ant:
    def __init__(self, n: int) -> None:
        # integer tour_length % the ant’s tour length
        self.tour_length = 0
        # integer tour[n þ 1] % ant’s memory storing (partial) tours
        self.tour = np.zeros((n + 1,))
        # integer/boolean visited[n] % visited cities
        self.visited = np.zeros((n,), dtype=bool)

    # visit node j
    def visit(self, j):
        self.visited[j] = True