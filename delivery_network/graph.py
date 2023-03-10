import graphviz

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
        self.edges = []
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            sorted_keys = sorted(self.graph.keys())
            for source in sorted_keys:
                output += f"{source}-->{self.graph[source]}\n"
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

        return None


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
                        return result
            return None
        return look_for_path(src,[src])
    #la complexité en temps est de l'ordre de O(n²) dans le pire des cas (graphe connecté) avec n le nombre de noeud

    ## Algorithme de Djikstra ##

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
        return path if path != [dest] else None
    #complexité en temps de n²ln(n) avec n le nombre de noeud dans le pire des cas(graphe connecté)
    
    def min_power(self, src, dest):
        # Trouver la puissance maximale du graphe
        max_puissance = max([edge[1] for node in self.nodes for edge in self.graph[node]])
        # Initialiser les bornes inférieure et supérieure de la recherche binaire
        lower_bound = 0
        upper_bound = max_puissance
        best_path = None
        # Recherche binaire pour trouver la puissance minimale requise
        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            path = self.get_shortest_path_with_power(src, dest, mid)
            if path is not None:
                best_path = path
                upper_bound = mid - 1
            else:
                lower_bound = mid + 1
        # Retourner le chemin et la puissance minimale requise
        return (best_path, lower_bound)
    #complexité de n²ln²(n) dans le pire des cas (graphe connecté)
   
    def graphic_representation(self, src, dest, power):

        repr = graphviz.Graph(comment = "Graphe non orienté", strict = True)
        if self.nb_nodes == self.nb_edges +1:
            path_min_power = self.min_power1(src,dest)[1]
        else:
            path_min_power = self.get_shortest_path_with_power(src, dest, power)
        graph = self.graph

        for node, neighbors in graph.items():
            repr.node(str(node), style = "filled", color = "lightblue")
            for neighbor, power_min, dist in neighbors:
                    repr.edge(str(node), str(neighbor), label="D : " + str(dist) + "\nP_min : " + str(power_min), constraint='true')

        if path_min_power != None:
            for node in path_min_power :
                repr.node(str(node), style = "filled", color = "red")
            for i in range(0, len(path_min_power)-1):
                node1 = path_min_power[i]
                node2 = path_min_power[i+1]

                repr.edge(str(node1), str(node2), label="Shortest path", color ="red", fontcolor ="red", constraint='true')

        repr.view()
        return None
    
    def get_edge(self, node1, node2, power_min):
        self.edges.append((node1, node2, power_min))

    def min_power1(self, src, dest):
        visited = {node : False for node in self.nodes}
        power_min = {node : -1 for node in self.nodes}
        power_min[src] = 0
        path = []

        def dfs(node, path):
            if node == dest:
                return power_min[dest], path
            for neighbor in self.graph[node]:
                neighbor, power, dist = neighbor
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if power_min[neighbor] < power:
                        power_min[neighbor] = power
                    result = dfs(neighbor, path + [neighbor])
                    if result is not None:
                        return result
            return None
        
        result = dfs(src, [src])
        if result is not None:
            for current_node in reversed(result[1]):
                path += [current_node]
                if current_node == src:
                    break
            return result[0], list(reversed(path))
        else:
            return None, []




"""
def graph_from_file(filename):
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min) # will add dist=1 by default
                g.get_edge(node1, node2, power_min)
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
                g.get_edge(node1, node2, power_min)
            else:
                raise Exception("Format incorrect")
    return g
"""
def graph_from_file(filename):
    f = open(filename, "r")
    with f as file:
        lines = f.readlines()
        line1 = lines.pop(0).split()
        g = Graph([i for i in range(1,int(line1[0])+1)])
        for i in range(len(lines)):
            lines[i] = lines[i].split()
            if len(lines[i]) == 3:
                node1, node2, power_min = lines[i]
                g.add_edge(int(node1), int(node2), int(power_min)) # will add dist=1 by default
                g.get_edge(int(node1), int(node2), int(power_min))
            elif len(lines[i]) == 4:
                node1, node2, power_min, dist = lines[i]
                g.add_edge(int(node1), int(node2), int(power_min), int(dist))
                g.get_edge(int(node1), int(node2), int(power_min))
            else:
                raise Exception("Format incorrect")
    f.close    
    return g

## Algorithme de Kruskal et structure Union-Find ##

#La complexité de l'algorithme de Kruskal est dominée par le tri des arêtes, qui prend O(E log E) temps, où E 
# est l'ensemble des arêtes de l'arbre couvrant de poids minimal créé (graphe connexe sans cycle).
# Ensuite, nous parcourons toutes les arêtes triées et effectuons une opération Union-Find sur chaque arête, 
# ce qui prend O(E alpha(V)) temps, où alpha(V) est la fonction inverse d'Ackermann et est essentiellement 
# constante pour des tailles de graphe pratiques. 
# En conséquence, la complexité totale de l'algorithme de Kruskal est de O(E log E) + O(E alpha(V)) = O(E log E).

def kruskal(graph):
    g_mst = Graph()
    g_mst.nodes = graph.nodes

    uf = UnionFind(graph.nodes)

    edges = sorted(graph.edges, key=lambda x: x[2])

    for edge in edges:
        node1, node2, weight = edge

        if uf.find(node1) != uf.find(node2):
            g_mst.add_edge(node1, node2, weight)
            uf.union(node1, node2)
    return g_mst

class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return
        if self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v
        elif self.rank[root_u] > self.rank[root_v]:
            self.parent[root_v] = root_u
        else:
            self.parent[root_v] = root_u
            self.rank[root_u] += 1

