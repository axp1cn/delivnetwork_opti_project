import graphviz

dot = graphviz.Graph(comment='Graphe non orienté', strict=True)

graphe = {'A': [('B', 2, 3), ('C', 3, 5)],
          'B': [('C', 1, 1), ('D', 4, 2)],
          'C': [('D', 1, 4)],
          'D': []}

for noeud in graphe.keys():
    dot.node(noeud)

for depart, voisins in graphe.items():
    for arrivee, distance, puissance in voisins:
        dot.edge(depart, arrivee, label='Distance : ' + str(distance) + '\nPuissance minimale : ' + str(puissance), constraint='true')

dot.view()