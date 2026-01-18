# Data Directory

This directory contains input files for the delivery network optimization project.

**Note**: Data files (`*.in`, `*.out`) are excluded from git. Users should provide their own data files or download them separately.

## File Formats

### Network Files (`network.*.in`)
Graph definitions with the following format:
- First line: `n m` (number of nodes and edges)
- Following `m` lines: `node1 node2 power_min [distance]`
  - `node1`, `node2`: Connected nodes
  - `power_min`: Minimum power required to traverse the edge
  - `distance`: Optional distance (defaults to 1 if not specified)

### Route Files (`routes.*.in`)
Delivery routes with profit information:
- First line: `T` (number of routes)
- Following `T` lines: `node1 node2 profit`
  - `node1`, `node2`: Origin and destination nodes
  - `profit`: Profit gained if route is covered

### Truck Files (`trucks.*.in`)
Truck catalog with specifications:
- First line: `K` (number of truck models)
- Following `K` lines: `power cost`
  - `power`: Truck power capacity
  - `cost`: Truck purchase cost

## Usage

These files are used by the optimization algorithms in `src/delivery_network/`. 

**Note**: If data files are large, consider adding them to `.gitignore` and providing instructions for users to download them separately, or use Git LFS for version control.

## Example

```python
from graph import graph_from_file

# Load a network
g = graph_from_file('data/network.1.in')

# Load routes and trucks (see graph.py for methods using these files)
```
