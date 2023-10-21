from __future__ import annotations

import dataclasses
import math
from queue import PriorityQueue
from typing import Callable, List, Optional, Tuple

from graph import Graph
from utils import euclidean_distance


@dataclasses.dataclass
class CurrentNode:
    """
    This dataclass represents the node that is reached during search.
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


def uniform_cost_search(
    g: Graph,
    source: str,
    dest: str,
    energy_budget: float = math.inf,
    verbose=True,
) -> Optional[Tuple[List[str], float, float]]:
    """
    Performs unifrom cost search to find the shortest path between
    source and destination in the graph.

    Args:
        g: Graph to perform search on
        source: source node
        dest: destination node
        verbose: whether to print search information

    Returns:
        the shortest path from source to destination and the distance
        returns None if there is no path found
    """
    shortest_distance = {node: math.inf for node in g.graph.keys()}
    shortest_distance[source] = 0
    least_energy_cost = {node: math.inf for node in g.graph.keys()}
    least_energy_cost[source] = 0

    nodes_searched = set()
    edges_searched = 0

    pq = PriorityQueue()
    pq.put(CurrentNode(source, 0, 0, [source]))

    while not pq.empty():
        cur_node = pq.get()

        if verbose:
            nodes_searched.add(cur_node.node)

        if cur_node.node == dest:
            if verbose:
                print(f"Number of nodes search: {len(nodes_searched)}")
                print(f"Number of edges search: {edges_searched}")
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

                if verbose:
                    edges_searched += 1

                pq.put(
                    CurrentNode(
                        next_node,
                        distance_to_next,
                        energy_cost_to_next,
                        cur_node.path + [next_node],
                    )
                )

    return None


@dataclasses.dataclass
class CurrentNodeWithHeuristic:
    """
    This dataclass represents the node that is reached during
    search with heuristics.
    """

    node: str
    total_distance: float
    distance_from_source: float
    energy_cost_from_source: float
    # we store the path here as with the energy budget, we may visit nodes via
    # more than 1 path and cannot backtrack without storing the entire path
    path: List[str]

    def __lt__(self, other: CurrentNodeWithHeuristic) -> bool:
        """
        Check if a CurrentNode is less than another CurrentNode.
        A node is searched first if the total distance from source is
        lower than another node. If both total distances are the same,
        energy cost is then compared.

        Args:
            other: the other node to compare to

        Returns:
            if this node should be less than the other node
        """
        if self.total_distance == other.total_distance:
            return self.energy_cost_from_source < other.energy_cost_from_source
        return self.total_distance < other.total_distance


def a_star_search(
    g: Graph,
    source: str,
    dest: str,
    energy_budget: float = math.inf,
    heuristic: Callable[
        [Tuple[float, float], Tuple[float, float]], float
    ] = euclidean_distance,
    heuristic_weight: float = 1,
    verbose: bool = True,
):
    """
    Performs A* search to find the shortest path between
    source and destination in the graph.

    Args:
        g: Graph to perform search on
        source: source node
        dest: destination node
        heuristic: heuristic function
        heuristic_weight: heuristic weight
        verbose: whether to print search information

    Returns:
        the shortest path from source to destination and the distance
        returns None if there is no path found
    """
    shortest_distance = {node: math.inf for node in g.graph.keys()}
    shortest_distance[source] = 0

    least_energy_cost = {node: math.inf for node in g.graph.keys()}
    least_energy_cost[source] = 0

    pq = PriorityQueue()
    pq.put(
        CurrentNodeWithHeuristic(
            source,
            heuristic(g.get_coord(source), g.get_coord(dest))
            * heuristic_weight,
            0,
            0,
            [source],
        )
    )

    nodes_searched = set()
    edges_searched = 0

    while not pq.empty():
        cur_node = pq.get()

        if verbose:
            nodes_searched.add(cur_node.node)

        if cur_node.node == dest:
            if verbose:
                print(f"Number of nodes search: {len(nodes_searched)}")
                print(f"Number of edges search: {edges_searched}")

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
                # even if there is already a path with lower total cost,
                # we still need to check a path with less energy cost
                or energy_cost_to_next < least_energy_cost[next_node]
            ) and energy_cost_to_next <= energy_budget:
                shortest_distance[next_node] = min(
                    distance_to_next, shortest_distance[next_node]
                )
                least_energy_cost[next_node] = min(
                    energy_cost_to_next, least_energy_cost[next_node]
                )

                if verbose:
                    edges_searched += 1

                heuristic_function_to_next = (
                    heuristic(g.get_coord(next_node), g.get_coord(dest))
                    * heuristic_weight
                )
                total_cost_to_next = (
                    distance_to_next + heuristic_function_to_next
                )
                pq.put(
                    CurrentNodeWithHeuristic(
                        next_node,
                        total_cost_to_next,
                        distance_to_next,
                        energy_cost_to_next,
                        cur_node.path + [next_node],
                    )
                )
    return None
