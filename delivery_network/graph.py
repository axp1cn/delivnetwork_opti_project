class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1
    

    def get_path_with_power(self, src, dest, power):
        visited = {node:False for node in self.nodes} 
        def look_for_path(node, path):
            if node==dest:
                return path
            for neighbor in self.graph[node]:
                neighbor, power_min, dist = neighbor
                if not visited[neighbor] and power_min<=power:
                    visited[neighbor] = True
                    result= look_for_path(neighbor, path+[neighbor])
                    if result is not None:
                        return(result)
            return None
        return look_for_path(src,[src])

    def get_shortest_path_with_power(self, src, dest, power):
        visited = []
        distances = {node: float('inf') for node in self.nodes}
        distances[src] = 0
        previous = {node: None for node in self.nodes}
        current_node = src
        while current_node != dest:
            if current_node is None:
                return None
            visited.append(current_node)
            for neighbor, power_min, edge_distance in self.graph[current_node]:
                if power_min > power or neighbor in visited:
                    continue
                new_distance = distances[current_node] + edge_distance
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
            unvisited = {node: distances[node] for node in distances if node not in visited}
            if not unvisited:
                return None
            current_node = min(unvisited, key=unvisited.get)
        path = []
        while current_node is not None:
            path.insert(0, current_node)
            current_node = previous[current_node]
        return path

    

    def connected_components(self):
        con_comps =[]
        visited = {node:False for node in self.nodes}

        def dfs(node):
            component = [node]
            for neighbor in self.graph[node]:
                neighbor = neighbor[0]
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component += dfs(neighbor)
            return component

        for node in self.nodes:
            if not visited[node]:
                visited[node] = True
                con_comps.append(dfs(node))
        
        return con_comps


    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def min_power(self, src, dest):
        return 'sophie <3'

def graph_from_file(filename):
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min) # will add dist=1 by default
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g

import graphviz 
