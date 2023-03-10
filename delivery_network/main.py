from graph import Graph, graph_from_file, kruskal
import graphviz, time, random, statistics

data_path = "/Users/axelpincon/Desktop/ENSAE/S2/Projet Python/projet_prog_ensae/python_project_afp/input/"
file_name = "network.1.in"

g = graph_from_file(data_path + file_name)
print(g.graph)
g_mst = kruskal(g)
g.graphic_representation(4,7,1100)

"""
# Question 10 : Calcul de la vitesse d'exécution du code développé dans la séance 1, en particulier la méthode min_power

for i in range (1,11):
    graph = graph_from_file(data_path + "routes."+str(i)+".in")
    counter = []
    for nb_tests in range(2):
        start_time = time.perf_counter()
        print(graph.min_power(random.randrange(1,graph.nb_nodes+1), random.randrange(1,graph.nb_nodes+1)))
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        counter.append(execution_time)
    average_speed = statistics.mean(counter)

    print("La vitesse d'exécution moyenne de la méthode min_power pour le fichier routes.{}.in est de {}.".format(i, round(average_speed,3)))

"""

