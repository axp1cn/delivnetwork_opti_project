from graph import Graph, graph_from_file, graphic_representation
import graphviz

data_path = "/Users/axelpincon/Desktop/ENSAE/S2/Projet Python/projet_prog_ensae/python_project_afp/input/"
file_name = "network.04.in"

g = graph_from_file(data_path + file_name)
print(g)

repr = graphviz.Graph(comment = "Graphe non orienté", strict = True)

for node, neighbors in g.items():
    repr.node(node, style = "filled", color = "lightblue")
    for neighbor, power_min, dist in neighbors:
        repr.edge(node, neighbor, label='D : ' + str(dist) + '\nP_min : ' + str(power_min), constraint='true')

repr.view()