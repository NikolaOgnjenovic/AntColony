import math
from typing import List

from ACO import AntColony

import numpy as np
import matplotlib.pyplot as plt


class City:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __iter__(self):
        return iter(self.__dict__.values())


def get_distance_matrix(cities: List[City]):
    """
    Generiše matricu udaljenosti između gradova.

    Args:
       cities (List[City]): Lista gradova.

    Returns:
       np.ndarray: Matrica udaljenosti.
    """
    distances = []
    for city in cities:
        arr = []
        for neighbour in cities:
            if city == neighbour:
                arr.append(np.inf)
            else:
                arr.append(math.sqrt((city.x - neighbour.x) ** 2 + (city.y - neighbour.y) ** 2))
        distances.append(arr)

    return np.array(distances)


def parse(filepath: str) -> List[City]:
    """
    Parsira datoteku sa podacima o gradovima.

    Args:
        filepath (str): Putanja do datoteke.

    Returns:
        List[City]: Lista gradova sa koordinatama.
    """
    cities = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = line.split(" ")
            cities.append(City(float(line[1]), float(line[2])))

        return cities


def plot(cities: List[City], shortest_path_indices: List[int], shortest_path_distance: float):
    """
    Generiše matricu udaljenosti između gradova.

    Args:
       cities (List[City]): Lista gradova.
       shortest_path_indices (List[int]): Lista indeksa gradova u najkraćem putu.
       shortest_path_distance (float): Distanca najkraćeg puta.

    Returns:
       np.ndarray: Matrica udaljenosti između gradova.
    """
    x_coords, y_coords = zip(*cities)

    # Prikazivanje gradova
    plt.scatter(x_coords, y_coords, color='red')

    # Prikazivanje najkrace putanje
    for i in range(len(shortest_path_indices) - 1):
        city1, city2 = shortest_path_indices[i], shortest_path_indices[i + 1]
        plt.plot([x_coords[city1], x_coords[city2]], [y_coords[city1], y_coords[city2]], color='blue')

    # Dodavanje labela
    plt.title(f'Najkraći put - Distanca: {shortest_path_distance}')
    plt.xlabel('X koordinata')
    plt.ylabel('Y koordinata')

    plt.show()


def run():
    # Parsiranje gradova i matrice udaljenosti
    cities = parse('data_tsp.txt')
    distances = get_distance_matrix(cities)

    # Pokretanje algoritma
    ant_colony = AntColony(distances, 1, 100, 0.95, alpha=1, beta=1, seed=42)
    (shortest_path_indices, shortest_path_distance) = ant_colony.run()

    # Prikazivanje resenja
    plot(cities, shortest_path_indices, shortest_path_distance)
    print(f"Ukupna distanca najkraćeg puta: {shortest_path_distance}")


if __name__ == '__main__':
    run()
