import math
from typing import List, Tuple

from ACO import AntColony

import numpy as np
import matplotlib.pyplot as plt


def get_distance_matrix(cities):
    distances = []
    for city in cities:
        arr = []
        for neighbours in cities:
            if city == neighbours:
                arr.append(np.inf)
            else:
                arr.append(math.sqrt((city[0] - neighbours[0]) ** 2 + (city[1] - neighbours[1]) ** 2))
        distances.append(arr)
    return np.array(distances)


def parse(filename: str) -> List[Tuple[float, float]]:
    cities = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = line.split(" ")
            cities.append((float(line[1]), float(line[2])))
        return cities


def plot(cities: List[Tuple[float, float]], shortest_path_cities: List[Tuple[int, int]]):
    x_coords, y_coords = zip(*cities)

    # Plotting the cities
    plt.scatter(x_coords, y_coords, color='red')

    # Plotting the shortest path
    for edge in shortest_path_cities:
        city1, city2 = edge
        plt.plot([x_coords[city1], x_coords[city2]], [y_coords[city1], y_coords[city2]], color='blue')

    # Adding labels
    plt.title(f'Shortest Path starting from {shortest_path_cities[0][0]}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Show the plot
    plt.show()


def run():
    cities = parse('mravinjak.txt')
    distances = get_distance_matrix(cities)

    ant_colony = AntColony(distances, 1, 100, 0.95, alpha=1, beta=1, seed=42)
    (shortest_path_cities, shortest_path_distance) = ant_colony.run()

    plot(cities, shortest_path_cities)
    print(f"Shortest path distance: {shortest_path_distance}")


if __name__ == '__main__':
    run()
