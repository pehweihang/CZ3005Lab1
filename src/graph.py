from __future__ import annotations

import json
from typing import Dict, List, Tuple


class Graph:
    graph: Dict[str, List[str]]
    coord: Dict[str, Tuple[float, float]]
    dist: Dict[str, float]
    cost: Dict[str, float]

    def __init__(
        self, graph_path: str, coord_path: str, cost_path: str, dist_path: str
    ) -> None:
        with open(graph_path) as f:
            self.graph = json.load(f)
        with open(coord_path) as f:
            self.coord = json.load(f)
        with open(cost_path) as f:
            self.cost = json.load(f)
        with open(dist_path) as f:
            self.dist = json.load(f)

    def __getitem__(self, node: str) -> List[str]:
        return self.graph[node]

    def __len__(self) -> int:
        return len(self.graph)

    def _edge_key(self, u: str, v: str) -> str:
        return f"{u},{v}"

    def nodes(self) -> List[str]:
        return [k for k in self.graph.keys()]

    def get_dist(self, u: str, v: str) -> float:
        return self.dist[self._edge_key(u, v)]

    def get_cost(self, u: str, v: str) -> float:
        return self.cost[self._edge_key(u, v)]

    def get_coord(self, node: str) -> Tuple[float, float]:
        return self.coord[node]
