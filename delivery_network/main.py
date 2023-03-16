from graph import Graph, graph_from_file, kruskal
import graphviz, time, random, statistics

data_path = "/Users/axelpincon/Desktop/ENSAE/S2/Projet Python/projet_prog_ensae/python_project_afp/input/"
file_name = "network.1.in"


g = graph_from_file(data_path + file_name)
g_mst = kruskal(g)
print(g_mst.min_power1(5,18))
print(g_mst.min_power2(5,18))

g_mst.graphic_representation(5,18,1000)


# Question 10 : Calcul de la vitesse d'exécution du code développé dans la séance 1, en particulier la méthode min_power
"""

for i in range (1,11):
    graph = graph_from_file(data_path + "network."+str(i)+".in")
    counter = []
    for nb_tests in range(2):
        a = random.randint(1,graph.nb_nodes)
        b = random.randint(1,graph.nb_nodes)
        while a == b:
            a = random.randint(1,graph.nb_nodes)
            b = random.randint(1,graph.nb_nodes)
        start_time = time.perf_counter()  
        print(graph.min_power1(a, b))
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        counter.append(execution_time)
    average_speed = statistics.mean(counter)

    print("La vitesse d'exécution moyenne de la méthode min_power pour le fichier routes.{}.in est de {}.".format(i, round(average_speed,3)))

"""

# Question 11 : 
"""Soient 1 et 2, deux noeuds de G.
Soit T le chemin entre les noeuds 1 et 2 dans A, de puissance minimale Tp.
La puissance minimale pour couvrir le chemin entre 1 et 2 dans G est Pp ayant pour chemin P.
Montrons par l'absurde que Tp=Pp, supposons Tp>Pp. (On a Pp<=Tp car A est inclu dans G)

Par conséquent, le chemin P doit contenir au moins une arête qui n'est pas dans T. Soit e cette arête, 
qui relie deux sommets u et v sur le chemin P. Cette arête a un poids inférieur au poids du sous chemin de T reliant u et v.
Maintenant, considérons l'arbre A'. Cet arbre est obtenu en remplaçant l'arête e par l'arête 
de A qui relie u et v.  
Par conséquent, le poids total de A' est inférieur ou égal au poids total de A. Cependant, A' est également un arbre couvrant de G. 
Cela contredit l'hypothèse selon laquelle A est un arbre couvrant de poids minimal dans G, car A' est également un arbre couvrant 
de poids minimal dans G. Nous en concluons donc que le chemin P doit être entièrement contenu dans T. 

"""


# Question 15 : Calcul de la vitesse d'exécution du code développé dans la séance 2, en particulier la méthode min_power2

for i in range (1,11):
    graph = graph_from_file(data_path + "network."+str(i)+".in")
    graph = kruskal(graph)
    counter = []
    for nb_tests in range(2):
        a = random.randint(1,graph.nb_nodes)
        print(a)
        b = random.randint(1,graph.nb_nodes)
        print(b)
        while a == b and a not in graph.graph and b not in graph.graph:
            a = random.randint(1,graph.nb_nodes)
            b = random.randint(1,graph.nb_nodes)
        start_time = time.perf_counter() 
        print(graph.min_power1(a, b))
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        counter.append(execution_time)
    average_speed = statistics.mean(counter)

    print("La vitesse d'exécution moyenne de la méthode min_power2 pour le fichier routes.{}.in est de {}.".format(i, round(average_speed,3)))


"""
for i in range (1,11):
    f = open(data_path + "routes."+str(i)+".in", "r")
    g = graph_from_file(data_path + "network."+str(i)+".in")
    g = kruskal(g)
    dest_path = "./input/routes."+str(i)+".out"
    y = open(dest_path, "w")
    lines = f.readlines()
    for j in range(1,len(lines)):
        lines[j] = lines[j].split()
        min_power, path = g.min_power2(int(lines[j][0]), int(lines[j][1]))
        y.write(str(min_power)+"\n")
    f.close()
    y.close()
"""