import argparse
import math
import random

from graph import Graph
from search import without_energy_budget
from utils import euclidean_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of shortest paths to sample",
        default=1000,
    )
    parser.add_argument("--seed", type=int, help="random seed", default=420)
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
    random.seed(args.seed)
    graph = Graph(
        **{k: v for k, v in vars(args).items() if k.endswith("path")}
    )
    max_gamma = math.inf
    nodes = graph.nodes()

    samples_processed = 0
    print(f"Sampling {args.num_samples} paths")
    while samples_processed < args.num_samples:
        source_node = nodes[random.randint(0, len(nodes) - 1)]
        dest_node = nodes[random.randint(0, len(nodes) - 1)]
        print(
            "Sampling path {} from {} to {} - current gamma: {}".format(
                samples_processed + 1, source_node, dest_node, max_gamma
            )
        )

        results = without_energy_budget.uniform_cost_search(
            graph, source_node, dest_node
        )
        if results is None:
            print(">>> No path found, trying another sample")
            continue

        _, distance = results
        euclidean_dist = euclidean_distance(
            graph.get_coord(source_node), graph.get_coord(dest_node)
        )
        max_gamma = min(max_gamma, distance / euclidean_dist)

        samples_processed += 1

    print(f"gama: {max_gamma}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
