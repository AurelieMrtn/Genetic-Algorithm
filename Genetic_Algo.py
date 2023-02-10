# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:01:54 2022

@author: aurel
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d

# Montrer les courbes

def Montrer(T,X,nv_X):
    t = np.array(T)
    x = np.array(X)
    nvx = np.array(nv_X)
    
    plt.scatter(t, x)
    plt.plot(t, x)
    plt.plot(t,nvx)
    
    plt.title('Y en fonction du temps')
    plt.xlabel("Temps")
    plt.ylabel("Valeurs")
    
    plt.show()

# Montrer l'orbite

def Orbite(X,Y):
    x = np.array(X)
    y = np.array(Y)
    
    plt.plot(x,y)
    
    plt.title('Orbite de Lissajous')
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.show()

# Fonction qui permet d'avoir une population aléatoire

def randomPop(x):
    Trois_pi = []
    for i in range (0,x):
        Trois_pi.append([random.uniform(-100,100),random.uniform(-100,100),random.uniform(-100,100)])
    return Trois_pi

# Fonction qui calcule les points avec des valeurs de pi ( Donne liste de x ou y )

def calculPts(p1,p2,p3,temps):
    pts_trouvés = []
    for t in temps:
        pts_trouvés.append(p1*sin(p2*t+p3))
    return pts_trouvés

# Fonction pour mesurer l'erreur entre les mesures et les valeurs calculées ( Donne liste de différences entre x et x caculé )

def mesureErreurs(pts_donnés, pts_trouvés):
    erreur = []
    for i in range(0, 29):
        erreur.append(pts_donnés[i] - pts_trouvés[i])
    return erreur

# Fitness pour avoir des coeffs pour la nouvelles génération (amplifie l'action de l'erreur)

def calculFitness(erreur):
    moy = 0
    for i in erreur:
        moy += abs(i)
    moy = moy / len(erreur)
    fit = 1 / moy
    return fit

# Fonction pour produire une génération(=pop + coef + nvx points) à partir d'une pop

def Generation(pop,liste):
    generation = [] # tableau avec les individus et leur coef fitness
    Tableau_de_X = [] # tableau avec les valeurs de X ou Y pour chaque individu
    for i in range(0,len(pop)):
        Xpts = calculPts(pop[i][0], pop[i][1], pop[i][2], T)
        erreur = mesureErreurs(liste,Xpts)
        generation.append([pop[i][0], pop[i][1], pop[i][2],calculFitness(erreur)])
        Tableau_de_X.append(Xpts)
        #Montrer(T, X, Xpts)
    #print('\n', len(generation), generation)
    return (generation,Tableau_de_X)

# Fonction pour avoir de nouvelles générations

def nvlGen(Gen, liste, nombre = 0):
    generation,Tableau_de_X = Gen
    
    if (nombre % 100 == 0):
        Montrer(T, liste, Tableau_de_X[0])
    
    nvlPop = []
    
    #selections des meilleurs pi avec les fitness
    
    taille = len(generation)
    generation = sorted(generation, key = lambda x: x[3], reverse = True)
    nvlPop = generation[0:taille//2]
    
     
    '''#Calcul de la moyenne des fitness
    moy = 0.0
    for i in range(0,len(generation)-1):
        moy += generation[i][3]
    moy = moy / len(generation)
    
    #Nouvelle population avec les fit > moy
    for i in range(0,len(generation)-1):
        if (generation[i][3] > moy) : nvlPop.append(generation[i][0:3])
    #print('\n Pop avant', len(nvlPop), nvlPop)'''
      
    
    #Utilisation de mutations et croisements aléatoires

    nb1 = random.randrange(0,len(generation)-len(nvlPop),1)
    
    # Mutation
    mutés = []
    for i in range(0,nb1-1):
        chromosome = generation[random.randrange(0,len(generation)-1)]
        chromosome = chromosome[0:3]
        a = random.randrange(0,3)
        b = random.randrange(0,3)
        pourcentA = chromosome[a]*5
        pourcentB = chromosome[b]*5
        chromosome[a] = random.uniform((chromosome[a]-pourcentA),(chromosome[a]+pourcentA))
        chromosome[b] = random.uniform((chromosome[b]-pourcentB),(chromosome[b]+pourcentB))
        mutés.append(chromosome)
    #print('\n mutés', len(mutés), mutés)
    
    nvlPop = nvlPop[:] + mutés[:]
    nb2 = len(generation) - len(nvlPop)
    
    # Pour que la population ne diminue pas pour ne pas disparaitre
    
    if (nb2 // 2 < nb2 / 2): nvlPop.append(nvlPop[0])
    
    # Croisemment
    croisés = []
    for i in range(0,nb2//2) :
        chromosome1 = generation[random.randrange(0,len(generation)-1)]
        chromosome1 = chromosome1[0:3]
        chromosome2 = generation[random.randrange(0,len(generation)-1)]
        chromosome2 = chromosome2[0:3]
        a = random.randrange(0,3)
        chromosome1[a], chromosome2[a] = chromosome2[a], chromosome1[a]
        croisés.append(chromosome1)
        croisés.append(chromosome2)
    #print('\n croisés', len(croisés), croisés)

    nvlPop = nvlPop[:] + croisés[:]
    #print('\n Pop', len(nvlPop), nvlPop)
    
    #Appliquer Generation à la nvl pop
    nvl_Generation,nvTab_X = Generation(nvlPop, liste)
    
    # Selectionne les pi avec le meilleur fitness
    
    cpt = 0
    for i in range(1, len(nvl_Generation)-1):
        if (nvl_Generation[i-1][3] < nvl_Generation[i][3]): cpt = i
        
    individu = nvl_Generation[cpt]
        
        
    if (nombre % 10 == 0):
        print("Génération n°", nombre, ":", individu, '\n')
        
    if (nombre == nb_generations_max or individu[3] > 24.7) : return "Génération n° " + str(nombre) + " : " + str(individu)
    
    return nvlGen(Generation(nvlPop, liste), liste, nombre + 1)

#### MAIN ####

population = 5000
nb_generations_max = 5000

# Valeurs mesurées

T = [0.765,1.001,1.282,1.362,1.387,1.729,1.926,1.981,2.088,2.108,2.31,2.369,2.749,2.832,3.171,3.577,3.952,4.138,4.17,
     4.366,4.469,4.90,4.941,4.99,5.02,5.086,5.129,5.179,5.278,6.162]

X = [-8.555,-11.957,-9.284,-8.398,-1.963,9.299,-12.848,-2.326,11.491,7.877,6.081,13.039,-4.161,-11.688,-2.683,
     11.521,-7.136,12.836,11.902,-1.653,11.697,-12.801,-7.01,6.074,11.835,7.809,-3.534,-12.795,3.441,1.058]

Y = [2.816,3.363,-17.443,15.592,-6.819,-22.37,10.488,9.046,17.269,22.782,-11.869,-3.768,5.828,-0.444,22.436,
     -16.296,20.189,-5.693,-22.908,4.865,-22.152,-8.255,-21.97,15.364,21.076,-22.896,3.823,18.315,-0.194,-22.345]

pop = randomPop(population)

# Pour X

#resX = (nvlGen(Generation(pop, X), X))
#print(resX)
# [13.19350859501943, -21.09927187458744, 1.1364096731102127, 23.765460290676582]

# Pour Y

resY = (nvlGen(Generation(pop, Y), Y))
print(resY)
# [-22.905955774647524, -41.09976408610216, 3.2902566402981224, 3.7761349954011028]

PtsX = calculPts(13.19350859501943, -21.09927187458744, 1.1364096731102127, T)
PtsY = calculPts(-22.905955774647524, -41.09976408610216, 3.2902566402981224, T)

orbite = Orbite(PtsX,PtsY)
#Montrer(T, Y, PtsY)
#Montrer(T,X, PtsX)