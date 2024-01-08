import math
from typing import List

from ACO import AntColony

import numpy as np
import matplotlib.pyplot as plt


class City:
    def __init__(self, index: str, x: float, y: float):
        self.index = index
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
            cities.append(City(str(line[0]), float(line[1]), float(line[2])))

        return cities


def plot(cities: List[City], shortest_path_indices: List[int], shortest_path_length: float, title: str):
    """
    Prikazuje na grafiku najkraći put koristeći date podatke.

    Args:
       cities (List[City]): Lista gradova.
       shortest_path_indices (List[int]): Lista indeksa gradova u najkraćem putu.
       shortest_path_length (float): Dužina najkraćeg puta.
       title (string): Naslov za grafik.
    """
    city_indices, x_coords, y_coords = zip(*cities)

    # Prikazivanje gradova
    plt.scatter(x_coords, y_coords, color='red')

    # Prikazivanje najkraće putanje
    
    for i in range(len(shortest_path_indices) - 1):
        city1, city2 = shortest_path_indices[i], shortest_path_indices[i + 1]
        plt.plot([x_coords[city1], x_coords[city2]], [y_coords[city1], y_coords[city2]], color='blue')
        plt.text(x_coords[city1], y_coords[city1], city_indices[city1], fontsize=8, ha='center', va='bottom', color='black')
    

    # Dodavanje labela
    plt.title(f'Najkraći put za {title} - dužina = {shortest_path_length}')
    plt.xlabel('X koordinata')
    plt.ylabel('Y koordinata')

    plt.show()


def run(n_ants: int, n_iterations: int, decay: float, alpha: float, beta: float, seed: int, title: str):
    # Parsiranje gradova i matrice udaljenosti
    cities = parse('data_tsp.txt')
    distances = get_distance_matrix(cities)

    # Pokretanje algoritma
    ant_colony = AntColony(distances, n_ants, n_iterations, decay, alpha=alpha, beta=beta, seed=seed)
    (shortest_path_indices, shortest_path_length) = ant_colony.run()

    # Prikazivanje rešenja
    plot(cities, shortest_path_indices, shortest_path_length, title)
    
    print(f"Najkraći put:")
    for index in shortest_path_indices:
        print(cities[index].index)
    print(f"Ukupna dužina najkraćeg puta: {shortest_path_length}")
    print("\n")


if __name__ == '__main__':
    # Default konfiguracija
    run(20, 100, 0.5, 1, 1, 42, "default konfiguraciju")

    # Najbolja konfiguracija (0.1 1 8 daje istu putanju)
    run(20, 100, 0.5, 1, 7, 42, "najbolju konfiguraciju")

    # Konfiguracija sa povecanim uticajem distance
    run(20, 100, 0.5, 1, 8, 42, "povecan uticaj distance")

    # Konfiguracija sa povecanim uticajem feromona
    run(20, 100, 0.5, 9, 1, 42, "povecan uticaj feromona")

    # Konfiguracija sa sporijim isparavanjem feromona
    run(20, 100, 0.1, 1, 1, 42, "sporije isparavanje feromona")

    # Konfiguracija sa brzim isparavanjem feromona
    run(20, 100, 0.9, 1, 1, 42, "brze isparavanje feromona")