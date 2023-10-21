import math
from typing import List, Tuple

from graph import Graph


def get_path_dist(graph: Graph, path: List[str]) -> float:
    """
    Calculate the distance of a path

    Args:
        graph: the graph the path is on
        path: path to calculate distance

    Returns:
        distance of path
    """
    dist = 0
    for i in range(len(path) - 1):
        dist += graph.get_dist(path[i], path[i + 1])
    return dist


def get_path_energy_cost(graph: Graph, path: List[str]) -> float:
    """
    Calculate the energy cost of a path

    Args:
        graph: the graph the path is on
        path: path to calculate energy cost

    Returns:
        energy_cost of path
    """
    energy_cost = 0
    for i in range(len(path) - 1):
        energy_cost += graph.get_cost(path[i], path[i + 1])
    return energy_cost


def euclidean_distance(
    x: Tuple[float, float], y: Tuple[float, float]
) -> float:
    """
    calculate the euclidean distance between two points

    Args:
        x: first point
        y: second point

    Returns:
        euclidean distance between the two points
    """
    return math.sqrt(abs(x[0] - y[0]) ** 2 + abs(x[1] - y[1]) ** 2)
