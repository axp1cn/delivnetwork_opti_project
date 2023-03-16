import graphviz, math

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
        "on ajoute des attributs nécessaires à la question 16"
        self.max_ancestors = dict()
        self.weight = dict()
        self.depth = dict()
        self.profondeur = dict()
        self.parents = []
        self.root = 2
    

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
    
    #QUESTION 1 (la fonction graph_from_file est sous la classe Graph):
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
        """on va créer une fonction récursive recherche en profondeur qui quand on lui donne un noeud va retourner la composante connexe 
        de ce noeud. Pour cela, on ajoute le noeud dans la composante, on parcourt tous les voisins du noeud, si il n'est pas 
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
        """on utilise notre fonction récursive DFS sur l'ensemble des noeuds non visités, chaque composante connexe est ajoutés dans 
        la liste des composantes connexes con_comps"""
        for node in self.nodes:
            if not visited[node]:
                visited[node] = True
                con_comps.append(dfs(node))
        
        return con_comps

    def connected_components_set(self):
        """on renvoie notre liste de liste con_comps comme un ensemble de frozensets qui sont immuables, on peut les utiliser comme clef de
        dictionnaire contrairement aux listes"""
        return set(map(frozenset, self.connected_components()))
    
    #QUESTION 3:
    """On va créer la méthode qui etant donnés un noeud source et un noeud destination ainsi qu'une puissance p donnée va déterminer si 
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
            "on sélectionne le noeud non visité avec la distance la plus courte comme noeud courant"
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
        return (lower_bound, best_path)
    "complexité de O(log(max_puissance)*Elog(V)) "

    #QUESTION 7:
    """On implémente une méthode de représentation graphique en utilisant le module graphviz, on affiche en rouge le chemin ayant 
    la puissance minimale qu'on a trouvé à la question 6"""
    def graphic_representation(self, src, dest, power):
        repr = graphviz.Graph(comment = "Graphe non orienté", strict = True)
        """si le graphe n'a pas de cycle et est connexe (arbre) on utilise le min_power2 du TD 2, si le graphe n'est pas 
        un arbre on utilise min_power de la question 6"""
        if self.nb_nodes == self.nb_edges +1:
            path_min_power = self.min_power2(src,dest)[1]
        
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

    #QUESTION 14: 
    """On va créer une méthode qui étant donnés un noeud source et un noeud destination va déterminer la puissance minimale requise 
    d'un camion pouvant couvrir le trajet entre ces deux noeuds ainsi que le chemin associé (sous forme de liste de noeuds).
    SPOILER : Cette méthode n'est pas adaptée en fin de compte (complexité plus grande que la méthode introduite juste en dessous)"""
    def min_power1(self, src, dest):
        """on crée un dictionnaire visited qui permet de tracer si un noeuf a déjà été visité ou non en initialisant la valeur associée 
        à chaque clé (noeud) à False"""
        visited = {node : False for node in self.nodes}
        """on crée un dictionnaire power_min qui permet de suivre la puissance minimale requise entre le noeud source et la clé (noeud)
        en initialisant la valeur associée à chaque clé (noeud) à -1 (cette valeur est arbitraire mais on la veut négative pour pour
        mettre à jour la puissance minimale requise dans la fonction récursive dfs) et celle du noeud source à 0"""
        power_min = {node : -1 for node in self.nodes}
        power_min[src] = 0
        path = []

        """on crée une fonction récursive dfs à l'image de celle créée lors de la première séance permettant de parcourir récursivement les
        voisins du noeud source afin de trouver le chemin qui sépare le noeud source du noeud de destinantion (on rappelle que ce chemin est
        unique et de poids (puissance requise) minimal puisque cette méthode va être appliquée à un arbre couvrant de poids minimal généré
        lors de la question précédente), ainsi que la puissance minimale pour réaliser ce trajet"""
        def dfs(node, path):
            if node == dest:
                return power_min, path
            for neighbor in self.graph[node]:
                neighbor, power, dist = neighbor
                if not visited[neighbor]:
                    visited[neighbor] = True
                    """c'est ici que l'initialisation de power_min à -1 (valeurs du dico) prend son sens, ainsi lorsque qu'un noeud est 
                    visité pour la première fois la valeur de power_min est mise à jour puisque la valeur de power est forcément plus grande"""
                    if power_min[neighbor] < power:
                        power_min[neighbor] = power
                    result = dfs(neighbor, path + [neighbor])
                    if result is not None:
                        return result
            return None
        
        "on récupère le résultat de notre fonction récursive"
        result = dfs(src, [src])
        """le chemin trouvé par notre fonction dfs est une suite de chemins partant du noeud source, jusqu'à l'arrivée au noeud destination,
        le chemin qui nous intéresse se trouve ainsi à la fin de la liste, donc on inverse la liste (chemin) avec la fonction reversed puis
        on récupère les dernières valeurs jusqu'au noeud source, puis on réinverse la liste"""
        if result is not None:
            for current_node in reversed(result[1]):
                path += [current_node]
                if current_node == src:
                    break
            return max(result[0][u] for u in path), list(reversed(path))
        else:
            return None, []
    
    """on veut une complexité en O(V), on a un arbre couvrant minimal min_tree, on se contente de trouver le chemin entre 
     src et dest (qui est unique et de poids minimal) et d'avoir sa puissance minimale """
    
    #QUESTION 14 CORRIGEE:

    def dfs14(self, root=None, profondeur=None, parents=None, visited=None, index=0 ):
        if visited is None and profondeur is None and parents is None and root is None:
            visited=set()
            profondeur=dict()
            parents=[0 for i in self.nodes]
            root = self.root
            profondeur[root]=0
            parents[root]=root
        visited.add(root)
        index+=1
        for son in self.graph[root]:
            if son[0] not in visited:
                profondeur[son[0]]=index
                parents[son[0]]= root
                self.dfs14(son[0], profondeur, parents, visited, index)
            else:
                pass
        self.profondeur = profondeur
        self.parents = parents
        return None
   
    def min_power3(self, src, dest):
        node1=src
        node2=dest
        path1=[src]
        path2=[dest]
        while node1 not in path2 or node2 not in path1:
            if self.parents[node1] != node1:
                node1=self.parents[node1]
                path1.append(node1)
            if self.parents[node2] != node2:
                node2=self.parents[node2]
                path2.append(node2)
        print(path1)
        path=path1+list(reversed(path2))
        return path
    
    def min_power4(self,src,dest):
        path1 = []
        path2 = []
        if self.profondeur[src]>self.profondeur[dest]:
            node = src
            path2 = [dest]
            while self.profondeur[node]!= self.profondeur[dest]:
                path1.append(node)
                node = self.parents[node]
            path1.append(node)
            node1 = node
            node2 = dest 
            while node1 != node2:
                path1.append(self.parents[node1])
                node1 = self.parents[node1]
                path2.append(self.parents[node2])
                node2 = self.parents[node2]
        if self.profondeur[src]<=self.profondeur[dest]:
            node = dest
            path1 = [src]
            while self.profondeur[node]!= self.profondeur[src]:
                path2.append(node)
                node = self.parents[node]
            path2.append(node)
            node2 = node
            node1 = src 
            while node1 != node2:
                path1.append(self.parents[node1])
                node1 = self.parents[node1]
                path2.append(self.parents[node2])
                node2 = self.parents[node2]
        path1.pop(-1)
        path=path1+list(reversed(path2))
        "on cherche maintenant à déterminer la puissance minimale requise pour parcourir ce chemin"
        min_power = 0
        for i in range(len(path)-2):
            next_node = self.graph[path[i]].index([t for t in self.graph[path[i]] if t[0] == path[i+1]][0])
            next_node1 = self.graph[path[i+1]].index([t for t in self.graph[path[i+1]] if t[0] == path[i+2]][0])
            if self.graph[path[i]][next_node][1] < self.graph[path[i+1]][next_node1][1]:
                min_power = self.graph[path[i+1]][next_node1][1]
            else: 
                min_power = self.graph[path[i]][next_node][1]

        return min_power, path

    
    def min_power2 (self, src, dest):
    
        "on crée une fonction simple qui trouve le chemin unique entre deux noeuds, src et dest, de notre arbre couvrant de poids minimal"
    
        stack = [src]
        visited = set([src])
        parent = {src: None}
        power = {src : 0}
        power_min = 0

        while stack:
            node = stack.pop()
            "nous avons trouvé le nœud que nous recherchons, donc construisons le chemin !"
            if node == dest:
                path = []
                while node:
                    if power[node] > power_min:
                        power_min = power[node]
                    path.insert(0, node)
                    node = parent[node]
                return power_min, path

            for child in self.graph[node]:
                child , pow = child[0], child[1]
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
                    parent[child] = node
                    power[child] = pow

        "nous n'avons pas trouvé le nœud que nous recherchions, donc il n'y a pas de chemin"
        return None
    
        """
        Calcul puissance minimale trop complexe
        best_path = find_path(src, dest)
        "on cherche maintenant à déterminer la puissance minimale requise pour parcourir ce chemin"
        min_power = 0
        for i in range(len(best_path)-2):
            "cette méthode n'est sûrement pas la plus optimée et lisible nous nous en excusons, le but étant simplement de retrouver l'
            arête (tuple) liant les noeuds du best_path deux à deux pour en récupérer la puissance minimale requise"
            next_node = self.graph[best_path[i]].index([t for t in self.graph[best_path[i]] if t[0] == best_path[i+1]][0])
            next_node1 = self.graph[best_path[i+1]].index([t for t in self.graph[best_path[i+1]] if t[0] == best_path[i+2]][0])
            "on veut retenir la valeur de puissance maximale (puissance minimale d'un camion requise pour parcourir le chemin trouvé"
            if self.graph[best_path[i]][next_node][1] < self.graph[best_path[i+1]][next_node1][1]:
                min_power = self.graph[best_path[i+1]][next_node1][1]
            else: 
                min_power = self.graph[best_path[i]][next_node][1]
            
        return min_power, best_path
        """ 

    #QUESTION 16:
    
    
    """ Essai 1

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
    
    def binary_search_power(self, start, end, max_dist):
        # Recherche binaire pour trouver la puissance minimale pour couvrir le trajet entre start et end
        low = 0
        high = max([edge[1] for node in self.nodes for edge in self.graph[node]])

        while low < high:
            mid = (low + high) // 2
            if self.is_power_sufficient(start, end, mid, max_dist):
                high = mid
            else:
                low = mid + 1

        return high
    """

    """Essai 2"""

    def preprocess(self):
        max_depth = int(math.ceil(math.log2(self.nb_nodes)))
        max_ancestors = {u: [None] * max_depth for u in self.nodes}
        depth = dict()
        for u in self.nodes:
            depth[u] = 0
            for v, weight, dist in self.graph[u]:
                depth[v] = depth[u] + 1
        for u, v, weight in self.edges:
            self.weight[(u, v)] = weight
            self.weight[(v, u)] = weight
        for u in self.nodes:
            self.dfs(u, None, 0, 1, max_ancestors)
        self.max_ancestors = max_ancestors
        self.depth = depth

    def dfs(self, u, p, w, depth, max_ancestors):
        max_ancestors[u][0] = p
        max_depth = int(math.ceil(math.log2(self.nb_nodes)))
        for i in range(1, depth):
            max_ancestors[u][i] = max_ancestors.get(max_ancestors[u][i-1], [None]*max_depth)[i-1]
        for v, weight, dist in self.graph[u]:
            if v != p:
                self.dfs(v, u, weight, depth+1, max_ancestors)

    def lca(self, u, v, depth):
        if depth[u] < depth[v]:
            u, v = v, u
        for i in range(depth[u]-depth[v]):
            u = self.max_ancestors[u][i]
        if u == v:
            return u
        for i in range(depth[u]-1, -1, -1):
            if self.max_ancestors[u][i] != self.max_ancestors[v][i]:
                u = self.max_ancestors[u][i]
                v = self.max_ancestors[v][i]
        return self.max_ancestors[u][0]

    def query(self, u, v, depth):
        lca_uv = self.lca(u, v, depth)
        max_weight = float('-inf')
        for i in [u, v]:
            while i != lca_uv:
                max_weight = max(max_weight, self.weight[(i, self.max_ancestors[i][0])])
                i = self.max_ancestors[i][0]
        return max_weight

    def solve(self, queries):
        self.preprocess()
        results = []
        for query in queries:
            results.append(self.query(*query, self.depth, self.max_ancestors))
        return results

"""
Version initiale de la fonction graph_from_file

def graph_from_file(filename):
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(list(range(1, n+1))) #il y avait une erreur dans le code initial (absence de la fonction list())
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
    "Nouvelle version de la fonction graph_from_file car les distances du fichiers routes.10.in sont des floats"
    f = open(filename, "r")
    with f as file:
        n, m = map(int, f.readline().split())
        g = Graph(list(range(1, n+1)))
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].split()
            if len(lines[i]) == 3:
                node1, node2, power_min = lines[i]
                g.add_edge(int(node1), int(node2), int(power_min)) # will add dist=1 by default
                g.get_edge(int(node1), int(node2), int(power_min))
            elif len(lines[i]) == 4:
                node1, node2, power_min, dist = lines[i]
                g.add_edge(int(node1), int(node2), int(power_min), float(dist))
                g.get_edge(int(node1), int(node2), int(power_min))
            else:
                raise Exception("Format incorrect")
    f.close    
    return g


#QUESTION 12: Algorithme de Kruskal et structure Union-Find

"""On va créer une fonction kruskal qui permet de convertir un graphe non-orienté en arbre couvrant de poids minimal. Pour cela on utilise l'
algorithme de Kruskal ainsi que la structure de données appelée Union-Find qui permet de déterminer si l'ajout d'une arête aux arêtes
déjà prises crée un cycle (ce que l'on veut banir car on veut un arbre connexe sans cycle)."""
def kruskal(graph):
    "on crée un nouvel objet avec la classe Graph qu'on appelle g_mst qui servira à stocker l'arbre couvrant de poids minimal"
    g_mst = Graph()
    "on copie les noeuds de l'objet graph d'origine dans g_mst"
    g_mst.nodes = graph.nodes

    uf = UnionFind(graph.nodes)

    """on trie les arêtes du graphe par ordre croissant de poids (puissance, troisième élément de chaque tuple edge) à l'aide de la fonction 
    sorted et d'une fonction lambda"""
    edges = sorted(graph.edges, key=lambda x: x[2])

    """on parcout chaque arête triée, on vérifie si les deux noeuds de l'arête n'appartiennent pas au même ensemble à l'aide de la méthode
     find de notre classe UnionFind puis on ajoute l'arête à l'objet g_mst et on fusionne les deux ensembles à l'aide de la méthode union de
     notre classe UnionFind"""
    for edge in edges:
        node1, node2, weight = edge

        if uf.find(node1) != uf.find(node2):
            g_mst.add_edge(node1, node2, weight)
            uf.union(node1, node2)
    return g_mst

"On va créer la structure de données Union-Find"
class UnionFind:
    "on initialise deux dictionnaires, parent et rank, qui stockent respectivement les parents et les rangs de chaque nœud dans l'ensemble"
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
    "on crée une méthode find qui permet de trouver la racine de l'arbre contenant un nœud donné, de manière récursive"
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    "on crée une méthode union qui fusionne deux ensembles contenant les nœuds 1 et 2"
    def union(self, node1, node2):
        root_node1 = self.find(node1)
        root_node2 = self.find(node2)
        if root_node1 == root_node2:
            return
        """si les noeuds n'appartiennent pas au même ensemble, on fusionne les deux ensembles en mettant à jour le parent de la racine 
        ayant le plus petit rang pour qu'il pointe vers la racine ayant le plus grand rang; si les deux racines ont le même rang, on 
        choisit arbitrairement l'une d'entre elles comme nouvelle racine et on augment son rang de 1"""
        if self.rank[root_node1] < self.rank[root_node2]:
            self.parent[root_node1] = root_node2
        elif self.rank[root_node1] > self.rank[root_node2]:
            self.parent[root_node2] = root_node1
        else:
            self.parent[root_node2] = root_node1
            self.rank[root_node1] += 1

"""La complexité de l'algorithme de Kruskal est dominée par le tri des arêtes, qui prend O(E log E) temps, où E est l'ensemble des arêtes 
de l'arbre couvrant de poids minimal créé (graphe connexe sans cycle). Ensuite, nous parcourons toutes les arêtes triées et effectuons 
une opération Union-Find sur chaque arête, ce qui prend O(E alpha(V)) temps, où alpha(V) est la fonction inverse d'Ackermann et est 
essentiellement constante pour des tailles de graphe pratiques. En conséquence, la complexité totale de l'algorithme de Kruskal est 
de O(E log E) + O(E alpha(V)) = O(E log E)."""