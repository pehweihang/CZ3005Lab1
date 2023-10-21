from __future__ import annotations

import dataclasses
import math
from queue import PriorityQueue
from typing import Callable, Dict, List, Optional, Tuple

from graph import Graph


@dataclasses.dataclass
class CurrentNode:
    """
    This dataclass represents the node that is reached during the search.
    """

    node: str
    distance_from_source: float
    energy_cost_from_source: float
    # we store the path here as with the energy budget, we may visit nodes via
    # more than 1 path and cannot backtrack without storing the entire path
    path: List[str]

    def __lt__(self, other: CurrentNode) -> bool:
        """
        Check if a CurrentNode is less than another CurrentNode.
        A node is searched first if the distance from source is
        lower than another node. If both distances are the same,
        energy cost is then compared.

        Args:
            other: the other node to compare to

        Returns:
            if this node should be less than the other node
        """
        if self.distance_from_source == other.distance_from_source:
            return self.energy_cost_from_source < other.energy_cost_from_source
        return self.distance_from_source < other.distance_from_source

@dataclasses.dataclass
class CurrentNode_astar:
    """
    This dataclass represents the node that is reached during the search.
    """

    node: str
    total_cost: float
    distance_from_source: float
    energy_cost_from_source: float
    # we store the path here as with the energy budget, we may visit nodes via
    # more than 1 path and cannot backtrack without storing the entire path
    path: List[str]

    def __lt__(self, other: CurrentNode_astar) -> bool:
        """
        Check if a CurrentNode is less than another CurrentNode.
        A node is searched first if the distance from source is
        lower than another node. If both distances are the same,
        energy cost is then compared.

        Args:
            other: the other node to compare to

        Returns:
            if this node should be less than the other node
        """
        if self.total_cost == other.total_cost:
            return self.energy_cost_from_source < other.energy_cost_from_source
        return self.total_cost < other.total_cost

def uniform_cost_search(
    g: Graph, source: str, dest: str, energy_budget: float = math.inf
) -> Optional[Tuple[List[str], float, float]]:
    """
    Performs unifrom cost search to find the shortest path between
    source and destination in the graph.

    Args:
        g: Graph to perform search on
        source: source node
        dest: destination node

    Returns:
        the shortest path from source to destination and the distance
        returns None if there is no path found
    """
    shortest_distance = {node: math.inf for node in g.graph.keys()}
    shortest_distance[source] = 0
    least_energy_cost = {node: math.inf for node in g.graph.keys()}
    least_energy_cost[source] = 0

    pq = PriorityQueue()
    pq.put(CurrentNode(source, 0, 0, [source]))

    while not pq.empty():
        cur_node = pq.get()

        if cur_node.node == dest:
            return (
                cur_node.path,
                cur_node.distance_from_source,
                cur_node.energy_cost_from_source,
            )

        for next_node in g[cur_node.node]:
            distance_to_next = cur_node.distance_from_source + g.get_dist(
                cur_node.node, next_node
            )
            energy_cost_to_next = (
                cur_node.energy_cost_from_source
                + g.get_cost(cur_node.node, next_node)
            )
            if (
                distance_to_next < shortest_distance[next_node]
                # even if there is already a path with shorter distance,
                # we still need to check a path with less energy cost
                or energy_cost_to_next < least_energy_cost[next_node]
            ) and energy_cost_to_next <= energy_budget:
                shortest_distance[next_node] = min(
                    distance_to_next, shortest_distance[next_node]
                )
                least_energy_cost[next_node] = min(
                    energy_cost_to_next, least_energy_cost[next_node]
                )
                pq.put(
                    CurrentNode(
                        next_node,
                        distance_to_next,
                        energy_cost_to_next,
                        cur_node.path + [next_node],
                    )
                )      

    return None


def _euclidean_distance(
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


def a_star_search(
    g: Graph,
    source: str,
    dest: str,
    energy_budget: float = math.inf,
    heuristic: Callable[
        [Tuple[float, float], Tuple[float, float]], float
    ] = _euclidean_distance,
    heuristic_weight: float = 1,
):
    total_cost_to_node = {node: math.inf for node in g.graph.keys()}
    total_cost_to_node[source] = heuristic(g.get_coord(source),g.get_coord(dest)) * heuristic_weight
    least_energy_cost = {node: math.inf for node in g.graph.keys()}
    least_energy_cost[source] = 0
    pq = PriorityQueue()
    pq.put(CurrentNode_astar(source, heuristic(g.get_coord(source),g.get_coord(dest)) * heuristic_weight, 0, 0,[source]))

    while not pq.empty():
        cur_node = pq.get()
        if cur_node.node == dest:
            return (
                cur_node.path,
                cur_node.distance_from_source,
                cur_node.energy_cost_from_source,
            )

        for next_node in g[cur_node.node]:
            distance_to_next = cur_node.distance_from_source + g.get_dist(
                cur_node.node, next_node
            )
            
            heuristic_function_to_next = (heuristic(g.get_coord(next_node),g.get_coord(dest)) * heuristic_weight)

            total_cost_to_next = distance_to_next + heuristic_function_to_next
            
            energy_cost_to_next = (
                cur_node.energy_cost_from_source
                + g.get_cost(cur_node.node, next_node)
            )
            if (
                total_cost_to_next < total_cost_to_node[next_node]
                # even if there is already a path with lower total cost,
                # we still need to check a path with less energy cost
                or energy_cost_to_next < least_energy_cost[next_node]
            ) and energy_cost_to_next <= energy_budget:
                total_cost_to_node[next_node] = min(
                    total_cost_to_next, total_cost_to_node[next_node]
                )
                least_energy_cost[next_node] = min(
                    energy_cost_to_next, least_energy_cost[next_node]
                )
                pq.put(
                    CurrentNode_astar(
                        next_node,
                        total_cost_to_next,
                        distance_to_next,
                        energy_cost_to_next,
                        cur_node.path + [next_node],
                    )
                )      
    return None