# CZ3005 Lab 1
This code solves the shortest traverse path between tow nodes within a certain energy budget.
## Requirments
```
python=3.11.5
```
## Usage
To run the code, run:
```bash
cd src
python main.py
```
### Parameters:
```bash
python main.py -h
```
```
usage: main.py [-h] [--heuristic-weight HEURISTIC_WEIGHT] [--graph-path GRAPH_PATH] [--coord-path COORD_PATH] [--cost-path COST_PATH] [--dist-path DIST_PATH] [source] [dest] [energy_budget]

positional arguments:
  source                source node
  dest                  destination node
  energy_budget         energy budget

options:
  -h, --help            show this help message and exit
  --heuristic-weight HEURISTIC_WEIGHT
                        weight of heuristic function
  --graph-path GRAPH_PATH
                        path to graph json
  --coord-path COORD_PATH
                        path to coord json
  --cost-path COST_PATH
                        path to cost json
  --dist-path DIST_PATH
                        path to dist json

```
