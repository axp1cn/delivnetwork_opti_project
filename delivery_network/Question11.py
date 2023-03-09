#Nous allons prouver que la puissance minimale pour couvrir un trajet t dans le graphe G est égale
#à la puissance minimale pour couvrir ce trajet t dans le graphe A. 

#Supposons que la puissance minimale pour couvrir un trajet t dans le graphe G est P. 
#Cela signifie qu'il existe un ensemble d'arêtes E dans G tel que la somme des poids des arêtes 
#dans E est égale à P et que ce ensemble E couvre le trajet t. Puisque A est un arbre couvrant de 
#poids minimal dans G, la somme des poids des arêtes dans A est minimale parmi tous les arbres 
#couvrants de G. De plus, puisque A est un arbre, il ne contient aucun cycle.

#Maintenant, considérons le graphe formé en ajoutant les arêtes dans E à l'arbre A. 
#Cela donne un sous-graphe connexe de G qui couvre également le trajet t. De plus, ce sous-graphe 
#contient exactement V arêtes, car l'arbre A en contient déjà V-1 et E en contient les autres.

#Puisque l'arbre A est sans cycle, il ne peut y avoir de cycle dans le sous-graphe formé en ajoutant
#E à A. Par conséquent, il existe une arête dans E qui relie deux sommets dans l'arbre A mais 
#qui n'est pas déjà dans A. Supposons que cette arête ait un poids de w.

#Maintenant, nous pouvons former un nouvel ensemble d'arêtes E' en remplaçant cette arête dans E 
#par le chemin le plus court dans A entre les deux sommets qu'elle relie. Cela donnera un ensemble 
#d'arêtes de même cardinalité que E mais dont la somme des poids est strictement inférieure à celle
#de E. En effet, le poids de E' sera égal au poids de E moins w, plus le poids du chemin le plus 
#court dans A entre les deux sommets qu'elle relie, et ce poids est strictement inférieur à w.

#Par conséquent, nous avons trouvé un ensemble d'arêtes E' qui couvre le trajet t et dont la somme 
#des poids est strictement inférieure à celle de E, ce qui contredit l'hypothèse selon laquelle P 
#est la puissance minimale pour couvrir le trajet t dans G. Ainsi, notre hypothèse de départ est 
#fausse et la puissance minimale pour couvrir un trajet t dans le graphe G est égale à la puissance 
#minimale pour couvrir ce trajet t dans le graphe A.