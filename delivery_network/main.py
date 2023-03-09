from graph import Graph, graph_from_file
import graphviz

data_path = "/Users/axelpincon/Desktop/ENSAE/S2/Projet Python/projet_prog_ensae/python_project_afp/input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g.graph)
print(g.get_shortest_path_with_power(1,4,10))
g.graphic_representation(1,4,10)