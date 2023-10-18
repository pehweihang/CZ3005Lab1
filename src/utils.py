from typing import List

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
