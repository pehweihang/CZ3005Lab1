from __future__ import annotations

import dataclasses
import math
from queue import PriorityQueue
from typing import Dict, List, Optional, Tuple

from graph import Graph


@dataclasses.dataclass
class CurrentNode:
    """
    This dataclass represents the node that is reached during the search.
    """

    node: str
    distance_from_source: float

    def __lt__(self, other: CurrentNode) -> bool:
        """
        Check if a CurrentNode is less than another CurrentNode.
        A node is searched first if the distance from source is
        lower than another node.

        Args:
            other: the other node to compare to

        Returns:
            if this node should be less than the other node
        """
        return self.distance_from_source < other.distance_from_source


def uniform_cost_search(
    g: Graph, source: str, dest: str
) -> Optional[Tuple[List[str], float]]:
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
    prev: Dict[str, Optional[str]] = {node: None for node in g.graph.keys()}

    pq = PriorityQueue()
    pq.put(CurrentNode(source, 0))

    while not pq.empty():
        cur_node = pq.get()

        if cur_node.node == dest:
            # find shortest path
            shortest_path = []
            p = dest
            while p is not None:
                shortest_path.append(p)
                p = prev[p]

            shortest_path.reverse()
            return shortest_path, cur_node.distance_from_source

        for next_node in g[cur_node.node]:
            distance_to_next = cur_node.distance_from_source + g.get_dist(
                cur_node.node, next_node
            )
            if distance_to_next < shortest_distance[next_node]:
                prev[next_node] = cur_node.node
                shortest_distance[next_node] = distance_to_next
                pq.put(CurrentNode(next_node, distance_to_next))

    return None
