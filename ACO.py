import numpy as np
from numpy.random import choice as np_choice
from typing import List, Tuple, Union


class AntColony(object):

    def __init__(self, distances: np.ndarray, n_ants: int, n_iterations: int, decay: float, alpha: Union[int, float] = 1, beta: Union[int, float] = 1, seed: float = 42):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponent on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta gives distance more weight. Default=1

        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)
        """
        self.distances: np.ndarray = distances
        self.pheromone: np.ndarray = np.ones(self.distances.shape) / len(distances)
        self.all_inds: range = range(len(distances))
        self.n_ants: int = n_ants
        self.n_iterations: int = n_iterations
        self.decay: float = decay
        self.alpha: Union[int, float] = alpha
        self.beta: Union[int, float] = beta
        np.random.seed(seed)

    def run(self) -> Tuple[List[Tuple[int, int]], float]:
        all_time_shortest_path: Tuple[List[Tuple[int, int]], float] = ([(-1, -1)], np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone = self.pheromone * self.decay
        return all_time_shortest_path

    def spread_pheromone(self, all_paths: List[Tuple[List[Tuple[int, int]], float]]) -> None:
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path: List[Tuple[int, int]]) -> float:
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self) -> List[Tuple[List[Tuple[int, int]], float]]:
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start: int) -> List[Tuple[int, int]]:
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # going back to where we started
        return path

    def pick_move(self, pheromone: np.ndarray, dist: float, visited: set) -> int:
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move
