import numpy as np
from numpy.random import choice as np_choice
from typing import List, Tuple


class AntColony(object):
    def __init__(self, distances: np.ndarray, n_ants: int, n_iterations: int, decay: float,
                 alpha: float = 1.0, beta: float = 1.0, seed: float = 42):
        """
        Inicijalizuje AntColony objekat.

        Args:
            distances (2D numpy.array): Matrica udaljenosti. Na dijagonali je pretpostavljena vrednost np.inf.
            n_ants (int): Broj mrava koji se koriste po iteraciji.
            n_iterations (int): Broj iteracija.
            decay (float): Vrednost raspada feromona. Uticaj feromona je se množi sa (1 - decay) svaku iteraciju.
            alpha (int or float): Eksponent za feromone, veće alfa daje veći uticaj feromona. Default=1
            beta (int or float): Eksponent za udaljenost, veće beta daje veći uticaj udaljenosti. Default=1
            seed (float): Seed za generisanje nasumičnih brojeva. Default=42
        """
        self.distances: np.ndarray = distances
        self.pheromone: np.ndarray = np.ones(self.distances.shape) / len(distances)
        self.all_indices: range = range(len(distances))
        self.n_ants: int = n_ants
        self.n_iterations: int = n_iterations
        self.decay: float = decay
        self.alpha: float = alpha
        self.beta: float = beta
        np.random.seed(seed)

    def run(self) -> Tuple[List[int], float]:
        """
        Pokreće algoritam optimizacije.

        Returns:
            Tuple: Torka stvari koja sadrži sortiran niz indeksa gradova obiđenih u najkraćem putu i pređenu distancu.
        """
        all_time_shortest_path: Tuple[List[int], float] = ([-1], np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone = self.pheromone * (1 - self.decay)
        return all_time_shortest_path

    def spread_pheromone(self, all_paths: List[Tuple[List[int], float]]) -> None:
        """
        Ažurira nivoe feromona na putanjama koje su pređene od strane mrava.

        Args:
            all_paths (List[Tuple[List[int], float]]): Lista putanja i njihovih udaljenosti.
        """
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_all_paths(self) -> List[Tuple[List[int], float]]:
        """
        Generiše putanje za sve mrave.

        Returns:
            List[Tuple[List[int], float]]: Lista putanja i njihovih udaljenosti.
        """
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start: int) -> List[int]:
        """
        Generiše putanju za pojedinačnog mrava.

        Args:
            start (int): Početna tačka za mrava.

        Returns:
            List[int]: Lista indeksa gradova koji predstavljaju putanju.
        """
        path = [start]
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append(move)
            prev = move
            visited.add(move)
        return path

    def gen_path_dist(self, path: List[int]) -> float:
        """
        Računa ukupnu udaljenost zadate putanje.

        Args:
            path (List[int]): Lista indeksa gradova koji predstavljaju putanju.

        Returns:
            float: Ukupna udaljenost puta.
        """
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i], path[i + 1]]
        return total_dist

    def pick_move(self, pheromone: np.ndarray, dist: float, visited: set) -> int:
        """
        Bira sledeći korak za mrava na osnovu nivoa feromona i udaljenosti.

        Args:
            pheromone (np.ndarray): Nivoi feromona na potezima.
            dist (float): Matrica udaljenosti.
            visited (set): Skup posećenih poteza.

        Returns:
            int: Indeks grada koji mrav obilazi u sledećem koraku.
        """
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_indices, 1, p=norm_row)[0]
        return move
