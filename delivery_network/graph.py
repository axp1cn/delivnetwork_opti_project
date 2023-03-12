import graphviz

class Graph:
    """
    On crée la classe Graph qui va définir un nouvel objet: les graphes, et contenir des méthodes qui nous seront utiles pour travailler sur
    ces graphes.
    Nous utiliserons ce cette classes les différentes variables ci dessous: 
    -----------
    nodes: NodeType
        Une liste de noeuds qui peuvent être de n'importe quels types.
        Nos fichiers network ont leurs noeuds comme entiers.
    graph: dict
        Un dictionnaire qui contient la liste d'adjacence de chaque noeud sous la forme
        graph[node] = [(neighbor1, p1, d1), (neighbor2, p2, d2), ...]
        avec p1, d1 la puissance minimale et la distance de l'arête (node, neighbor1) 
    nb_nodes: int
        Le nombre de noeud.
    nb_edges: int
        Le nombre d'arête. 
    """

    def __init__(self, nodes=[]):
        """
        Initialise le graphe avec une liste de noeud en paramètre (vide par défaut). 
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
        self.edges = []
    

    def __str__(self):
        """Affiche le nombre de noeud et d'arêtes ainsi que le graphe: chaque ligne on a un noeud (classé par ordre croissant) 
        et sa liste d'adjacence"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            sorted_keys = sorted(self.graph.keys())
            for source in sorted_keys:
                output += f"{source}-->{self.graph[source]}\n"
        return output
    #QUESTION 1 (graph_from_file est sous la classe graph):
    def add_edge(self, node1, node2, power_min, dist=1):
        """ Le but de cette fonction est d'ajouter une arête au graphe, on s'en servira dans notre fonction graph_from_file qui créé un graphe
        à partir d'un fichier network. 
        On fait attention a bien mettre à jour nb_nodes et nb_edges, et on rajoute les noeuds à la liste nodes si ils n'y sont pas, ainsi que
        le voisin dans la liste d'adjacence """
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
    
    #QUESTION 2:
    """  Etant donné un graphe, on va créer dans cette méthode la liste des composantes connexes"""
    def connected_components(self):
        """ on crée une liste vide qu'on remplira avec nos composantes(qui sont aussi des listes), et un dictionnaire visited 
        avec pour clef un noeud et pour valeur un booléen désignant si le noeud a été visité ou non"""
        con_comps =[]
        visited = {node:False for node in self.nodes}
        """ on créer une fonction récursive recherche en profondeur qui quand on lui donne un noeud va retourner la composante connexe 
        de ce noeud. Pour cela, on ajoute le noeud dans la composante parcourt tous les voisins du noeud, si il n'est pas 
        déjà visité on marque comme visité et réitère la fonction sur le noeud voisin
        La complexité de DFS est O(V+E) soit O(V²) dans le pire des cas (graphe connecté: E=V²)"""
        def dfs(node):
            component = [node]
            for neighbor in self.graph[node]:
                neighbor = neighbor[0]
                if not visited[neighbor]:
                    visited[neighbor] = True
                    component += dfs(neighbor)
            return component
        """ On utilise notre fonction récursive DFS sur l'ensemble des noeuds non visités, chaque composante connexe est ajouté dans 
        la liste des comosantes connexes"""
        for node in self.nodes:
            if not visited[node]:
                visited[node] = True
                con_comps.append(dfs(node))
        
        return con_comps

    def connected_components_set(self):
        """ On renvoie notre liste de liste con_comps comme un ensemble de frozensets qui sont immuables, on peut les utiliser comme clef de
        dictionnaire contrairement aux listes"""
        return set(map(frozenset, self.connected_components()))
    
    #QUESTION 3:
    """ On créer la méthode qui etant donné un noeud source et un noeud destination ainsi qu'une puissance p donnée va déterminer si 
    un camion de puissance p donnée va pouvoir passer et renvoyer un chemin si possible """
    def get_path_with_power(self, src, dest, power):
        visited = {node:False for node in self.nodes} 
        """on crée la fonction récursive look_for_path qui étant donné un noeud et une liste de noeuds renvoie la liste de neouds à laquelle 
        on a ajouté le chemin entre le noeud et la destination finale"""
        def look_for_path(node, path):
            if node==dest:
                return path
            """après avoir vérifié si l'on n'était pas déjà arrivé à la destination finale on va parcourir les voisins du noeud non déjà visité
            et si la condition de puissance est verifiée on ajoute le voisin au chemin et relance la fonction avec le voisin et le chemin
            en paramètre """
            for neighbor in self.graph[node]:
                neighbor, power_min, dist = neighbor
                if not visited[neighbor] and power_min<=power:
                    visited[neighbor] = True
                    result= look_for_path(neighbor, path+[neighbor])
                    if result is not None:
                        return result
            return None
        """une fois look_for_path définie, get_ path_with_power sera juste la recherche d'un chemin en partant de src avec comme
        chemin de noeud déjà parcouru [src], si jamais il n'y a pas de chemin possible, la fonction ne retourne rien"""
        return look_for_path(src,[src])
    """la complexité en temps est de l'ordre de O(E) soit  O(V²) dans le pire des cas (graphe connecté)"""

    #QUESTION 5:
    """On va utiliser l'algorithme de DJIKISTRA pour trouver le chemin le plus court sous contrainte de la puissance du camion
    on a en paramètre un noeud source, un noeud destination et la puissance du camion"""
    def get_shortest_path_with_power(self, src, dest, power):
        """on crée la liste des noeuds visités, un dictionnaire distances la distance du chemin le plus court entre le nœud source 
        et chaque nœud du graphe, et un dictionnaire previous pour stocké noeud précédent de chaque noeud dans le chemin le plus court"""
        visited = []
        distances = {node: float('inf') for node in self.nodes}
        distances[src] = 0
        previous = {node: None for node in self.nodes}
        current_node = src
        while current_node != dest:
            "on vérifie si il existe un chemin"
            if current_node is None:
                return None
            visited.append(current_node)
            "on va parcourir les voisins de noeud actuel non déjà visité, vérifier que la condition de puissance est vérifiée "
            for neighbor, power_min, edge_distance in self.graph[current_node]:
                if power_min > power or neighbor in visited:
                    continue
                """on ajoute la distance entre le noeud actuel et un voisin à la distance totale à la distance du chemin et on compare
                à ce qu'on obtient avec les autres voisins, Si la distance totale est inférieure à la distance stockée dans le dictionnaire 
                distances pour le voisin, le code met à jour la distance et le nœud précédent pour le voisin"""
                new_distance = distances[current_node] + edge_distance
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
            "on crée un dictionnaire avec les noeuds non visité et leur distance la plus courte au noeud source"
            unvisited = {node: distances[node] for node in distances if node not in visited}
            if not unvisited:
                return None
            "On sélectionne le noeud non visité avec la distance la plus courte comme noeud courant"
            current_node = min(unvisited, key=unvisited.get)
        "on crée la liste des noeuds composant le chemin le plus court entre le noeud actuel et le noeud source"
        path = []
        while current_node is not None:
            path.insert(0, current_node)
            current_node = previous[current_node]
        return path if path != [dest] else None
    "complexité en temps de O(Eln(V)) donc O(V²ln(V)) dans le pire des cas(graphe connecté)"
    
    #QUESTION 6:
    def min_power(self, src, dest):
        "On cherche la puissance maximale sur le graphe"
        max_puissance = max([edge[1] for node in self.nodes for edge in self.graph[node]])
        "On initialise les bornes de la recherche Binaire"
        lower_bound = 0
        upper_bound = max_puissance
        best_path = None
        "On effectue la recherche binaire (dichotomie)"
        while lower_bound <= upper_bound:
            mid = (lower_bound + upper_bound) // 2
            path = self.get_shortest_path_with_power(src, dest, mid)
            if path is not None:
                best_path = path
                upper_bound = mid - 1
            else:
                lower_bound = mid + 1
        "on retourne le chemin ainsi que la puissance minimale requise"
        return (best_path, lower_bound)
    "complexité de O(log(max_puissance)*Elog(V)) "

    #QUESTION 7:
    """On implémente une méthode de représentation graphique en utilisant graphviz, on affiche en rouge le chemin ayant 
    la puissance minimale qu'on a trouvé à la question 6"""
    def graphic_representation(self, src, dest, power):
        repr = graphviz.Graph(comment = "Graphe non orienté", strict = True)
        """si le graphe n'a pas de boucle et est connexe (arbre) on utilise le min_power1 du TD 2, si le graphe n'est pas 
        un arbre on utilise min_power de la question 6"""
        if self.nb_nodes == self.nb_edges +1:
            path_min_power = self.min_power1(src,dest)[1]
        
        else:
            path_min_power = self.min_power(src, dest)[1]
        graph = self.graph
        "on affiche les noeuds et les arêtes"
        for node, neighbors in graph.items():
            repr.node(str(node), style = "filled", color = "lightblue")
            for neighbor, power_min, dist in neighbors:
                    repr.edge(str(node), str(neighbor), label="D : " + str(dist) + "\nP_min : " + str(power_min), constraint='true')
        "on affiche en rouge le chemin solution de la question 6"
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
        
    #QUESTION 16:
    
    """il faudrait une fonction qui trouve les ancêtres et les stockent dans un dictionnaire ancestors"""
    
    def min_power_for_path(self, start, end):
        # Trouver la liste des ancêtres communs entre start et end
        common_ancestors = set(self.ancestors[start]).intersection(self.ancestors[end])

        # Trier les ancêtres par ordre décroissant de puissance minimale
        sorted_ancestors = sorted(common_ancestors, key=lambda x: self.ancestors[start][x][0], reverse=True)

        # Parcourir la liste des ancêtres et trouver la puissance minimale pour chaque trajet
        min_power = float('inf')
        current_node = start
        for ancestor in sorted_ancestors:
            ancestor_power, ancestor_dist = self.ancestors[start][ancestor]
            path_power = self.binary_search_power(current_node, ancestor, ancestor_power, ancestor_dist)
            min_power = max(min_power, path_power)
            current_node = ancestor

        # Trouver la puissance minimale pour le dernier trajet
        path_power = self.binary_search_power(current_node, end, 0, 1)
        min_power = max(min_power, path_power)

        return min_power
    
    def is_power_sufficient(self, start, end, power, max_dist):
        # Vérifie si la puissance power est suffisante pour couvrir le trajet entre start et end
        current_node = start
        while current_node != end:
            next_node, next_power, next_dist = None, None, None
            for neighbor, neighbor_power, neighbor_dist in self.graph[current_node]:
                if neighbor == self.ancestors[current_node][0]:
                    next_node, next_power, next_dist = neighbor, neighbor_power, neighbor_dist
                    break
            if next_node is None or next_dist > max_dist or next_power < power:
                return False
            max_dist -= next_dist
            power -= next_dist
            current_node = next_node
        return True
    
    def binary_search_power(self, start, end, min_power, max_dist):
        # Recherche binaire pour trouver la puissance minimale pour couvrir le trajet entre start et end
        low = min_power
        high = max([edge[1] for node in self.nodes for edge in self.graph[node]])

        while low < high:
            mid = (low + high) // 2
            if self.is_power_sufficient(start, end, mid, max_dist):
                high = mid
            else:
                low = mid + 1

        return high



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
#QUESTION 1 ET 4:
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

