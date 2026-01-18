# Delivery Network Optimization

This repository implements a graph-based optimization for delivery network routing and truck allocation under power and budget constraints.

## What It Demonstrates

- **Graph algorithms**: Implementation of Kruskal's MST algorithm with Union-Find data structure for efficient minimum spanning tree computation
- **Path optimization**: Binary search-based minimal power path finding with Dijkstra's algorithm for shortest paths under power constraints
- **Tree-based optimization**: Preprocessing with DFS for O(V) minimal power queries on MSTs using LCA (Lowest Common Ancestor) approach
- **Combinatorial optimization**: Greedy knapsack algorithm for truck allocation maximizing profit under budget constraints
- **Realistic modeling**: Extension incorporating route failure probabilities and fuel costs proportional to distance
- **Visualization**: Graphviz-based network visualization with highlighted optimal paths and truck allocations

## Key Results

- Achieved O(V) complexity for minimal power queries on MSTs (vs O(E log V) on general graphs)
- Efficient greedy allocation algorithm for large-scale truck routing problems
- Realistic profit optimization accounting for route reliability and operational costs
- See `tests/` for validation results on networks of varying sizes

## Repository Layout

```
.
├── src/ # Core graph algorithms and optimization code
│   ├── graph.py          # Graph class with MST, path finding knapsack methods
│   ├── main.py           # Example usage and performance benchmarks
├── data/                     # Network and route data files
│   ├── network.*.in          # Graph definitions (nodes, edges, power requirements)
│   ├── routes.*.in           # Delivery routes with profit values
│   └── trucks.*.in           # Truck catalog (power, cost)
├── tests/                    # Unit tests for all major algorithms
└── install_graphviz.sh       # Graphviz installation script

```

## Quickstart

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install graphviz for visualization (macOS)
brew install graphviz
# Or use the provided script
bash install_graphviz.sh
```

### Demo

```bash
# Run a simple example
python -c "
import sys
sys.path.append('src/delivery_network')
from graph import graph_from_file, kruskal

# Load network and compute MST
g = graph_from_file('data/network.1.in')
g_mst = kruskal(g)
g_mst.dfs14()

# Find minimal power path
power, path = g_mst.min_power4(1, 10)
print(f'Minimal power: {power}, Path: {path}')
"

# Run tests
python -m pytest tests/
```

## Method Overview

### Graph Representation
- Adjacency list representation with edge attributes (power requirement, distance)
- Support for weighted undirected graphs

### Minimal Power Path Finding
1. **General graphs**: Binary search over power values combined with Dijkstra's algorithm (O(E log V))
2. **MST optimization**: Preprocess tree with DFS to compute depths and parents, then use LCA approach for O(V) queries

### Truck Allocation (Knapsack)
- **Greedy algorithm**: Sort routes by profit/cost ratio, allocate trucks sequentially
- **Realistic extension**: Adjust profits by route failure probability `(1-ε)^(path_length)` and subtract fuel costs

### MST Construction
- Kruskal's algorithm with Union-Find data structure
- O(E log E) complexity for edge sorting and cycle detection

## Notes / Limitations

- Greedy knapsack provides approximate solutions; optimal solution would require dynamic programming (exponential complexity)


## References

- Kruskal, J. B. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. *Proceedings of the American Mathematical Society*, 7(1), 48-50.
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1(1), 269-271.
- Union-Find (Disjoint Set Union) data structure for efficient cycle detection
