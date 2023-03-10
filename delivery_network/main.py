from graph import Graph, graph_from_file, kruskal
import graphviz, time, random, statistics

data_path = "/Users/axelpincon/Desktop/ENSAE/S2/Projet Python/projet_prog_ensae/python_project_afp/input/"
file_name = "routes.1.in"


"""g = graph_from_file(data_path + file_name)
g_mst = kruskal(g)
print(g_mst)
print(g_mst.min_power1(random.randrange(1,g_mst.nb_nodes+1), random.randrange(1,g_mst.nb_nodes+1)))
print(g_mst.min_power1(random.randrange(1,g_mst.nb_nodes+1), random.randrange(1,g_mst.nb_nodes+1)))
print(g_mst.min_power1(1,19))"""

"""
# Question 10 : Calcul de la vitesse d'exécution du code développé dans la séance 1, en particulier la méthode min_power

for i in range (1,11):
    graph = graph_from_file(data_path + "routes."+str(i)+".in")
    counter = []
    for nb_tests in range(2):
        start_time = time.perf_counter()
        a = random.randint(1,graph.nb_nodes)
        b = random.randint(1,graph.nb_nodes)
        while a == b:
            a = random.randint(1,graph.nb_nodes)
            b = random.randint(1,graph.nb_nodes)  
        print(graph.min_power(a, b))
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        counter.append(execution_time)
    average_speed = statistics.mean(counter)

    print("La vitesse d'exécution moyenne de la méthode min_power pour le fichier routes.{}.in est de {}.".format(i, round(average_speed,3)))

"""


# Question 15 : Calcul de la vitesse d'exécution du code développé dans la séance 2, en particulier la méthode min_power1

for i in range (1,11):
    graph = graph_from_file(data_path + "routes."+str(i)+".in")
    graph = kruskal(graph)
    counter = []
    print (graph.nb_nodes)
    for nb_tests in range(2):
        start_time = time.perf_counter()
        a = random.randint(1,graph.nb_nodes)
        print(a)
        b = random.randint(1,graph.nb_nodes)
        print(b)
        while a == b:
            a = random.randint(1,graph.nb_nodes)
            b = random.randint(1,graph.nb_nodes)  
        print(graph.min_power1(a, b))
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        counter.append(execution_time)
    average_speed = statistics.mean(counter)

    print("La vitesse d'exécution moyenne de la méthode min_power1 pour le fichier routes.{}.in est de {}.".format(i, round(average_speed,3)))
