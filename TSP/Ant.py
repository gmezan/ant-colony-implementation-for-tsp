import numpy as np

# % Representation of ants
class Ant:
    def __init__(self, n: int) -> None:
        # double tour_length % the ant’s tour length
        self.tour_length = 0.0
        # integer tour_length % the ant’s tour length
        self.step = 0
        # integer tour[n þ 1] % ant’s memory storing (partial) tours
        self.tour = []
        # integer/boolean visited[n] % visited cities
        self.visited = np.zeros((n,), dtype=bool)
        # Flag to konw if ant has come back to initial node
        self.rounded = False

        # termination flag
        self.done = False

    # visit node j
    def visit(self, j, dist):
        self.tour.append(j)
        self.visited[j] = True
        if self.step > 0:
            self.tour_length += dist[self.tour[self.step - 1]][j]
        self.step += 1

    def reset(self):
        self.done = False
        self.step = 0
        self.tour_length = 0.0
        self.tour = []
        self.visited.fill(False)
        self.rounded = False

    def get_current_node(self):
        return self.tour[self.step - 1]

    def get_previous_node(self):
        if self.step <= 1:
            raise Exception('Bad execution: step still 1 after ant move')
        
        return self.tour[self.step - 2]