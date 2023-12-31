import argparse
import time

import search
from graph import Graph
from utils import get_path_dist, get_path_energy_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source", nargs="?", type=str, help="source node", default="1"
    )
    parser.add_argument(
        "dest", type=str, nargs="?", help="destination node", default="50"
    )
    parser.add_argument(
        "energy_budget",
        type=float,
        nargs="?",
        help="energy budget",
        default=287932,
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        help="weight of heuristic function",
        default=1.0299166238808808,
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        help="path to graph json",
        default="../data/G.json",
    )
    parser.add_argument(
        "--coord-path",
        type=str,
        help="path to coord json",
        default="../data/Coord.json",
    )
    parser.add_argument(
        "--cost-path",
        type=str,
        help="path to cost json",
        default="../data/Cost.json",
    )
    parser.add_argument(
        "--dist-path",
        type=str,
        help="path to dist json",
        default="../data/Dist.json",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    graph = Graph(
        **{k: v for k, v in vars(args).items() if k.endswith("path")}
    )

    print("Task 1")
    start_time = time.time()
    results = search.without_energy_budget.uniform_cost_search(
        graph, args.source, args.dest, verbose=True
    )
    end_time = time.time()
    if results is not None:
        shortest_path, distance = results
        print(f"Shortest path: {'->'.join(shortest_path)}")
        print(f"Shortest distance: {distance}")
        print(f"Time taken: {end_time - start_time}")
        assert distance == get_path_dist(graph, shortest_path)
    else:
        print(f"No path found from {args.source} to {args.dest}")
    print()

    print("Task 2")
    start_time = time.time()
    results = search.with_energy_budget.uniform_cost_search(
        graph,
        args.source,
        args.dest,
        energy_budget=args.energy_budget,
        verbose=True,
    )
    end_time = time.time()
    if results is not None:
        shortest_path, distance, energy_cost = results
        print(f"Shortest path: {'->'.join(shortest_path)}")
        print(f"Shortest distance: {distance}")
        print(f"Total energy cost: {energy_cost}")
        print(f"Time taken: {end_time - start_time}")
        assert distance == get_path_dist(graph, shortest_path)
        assert energy_cost == get_path_energy_cost(graph, shortest_path)
    else:
        print(f"No path found from {args.source} to {args.dest}")
    print()

    print("Task 3")
    start_time = time.time()
    results = search.a_star_search(
        graph,
        args.source,
        args.dest,
        energy_budget=args.energy_budget,
        heuristic_weight=args.heuristic_weight,
        verbose=True,
    )
    end_time = time.time()
    if results is not None:
        shortest_path, distance, energy_cost = results
        print(f"Shortest path length: {len(shortest_path)}")
        print(f"Shortest path: {'->'.join(shortest_path)}")
        print(f"Shortest distance: {distance}")
        print(f"Total energy cost: {energy_cost}")
        print(f"Time taken: {end_time - start_time}")
        assert distance == get_path_dist(graph, shortest_path)
        assert energy_cost == get_path_energy_cost(graph, shortest_path)
    else:
        print(f"No path found from {args.source} to {args.dest}")
    print()


if __name__ == "__main__":
    args = parse_args()
    main(args)
