# Delivery Network Optimization

Graph-based optimization for delivery routing and truck allocation under power and budget constraints.

## What’s inside
- **Minimum spanning tree (MST)** construction using **Kruskal + Union-Find**
- **Minimal-power routing**
  - On general graphs: **binary search** over feasible power + **Dijkstra** for shortest paths under the constraint
  - On the MST: **DFS preprocessing + LCA-style queries** for faster minimal-power path retrieval
- **Truck allocation under budget** via a **greedy knapsack-style heuristic** to maximize profit
- **More realistic modeling** with **route failure probabilities** and **fuel costs** proportional to distance
- **Optional visualization** using **Graphviz** to highlight selected paths and allocations

## Contributors
Axel Pincon, Matteo Alquier and Gabriel de Boerdere

## References
- Kruskal, Joseph B. (1956), *On the shortest spanning subtree of a graph and the traveling salesman problem*
- Dijkstra, Edsger W. (1959), *A note on two problems in connexion with graphs*

MIT license, feel free to use and adapt with attribution.
