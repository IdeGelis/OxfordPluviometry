# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:09:59 2018

@author: Iris and Mannaig
"""

import numpy as np
import matplotlib.pyplot as plt
from Reader import read

#import scipy as sc
#import lmfit

#def mod(t, A, omega,phi,cst):
#    return A*np.cos(omega*t + phi)+ cst
#    

    
if __name__=="__main__":
    nb_annee =25
    date_deb = 1900   
    temperature, tps = read("oxforddata.txt",date_deb,nb_annee)
    
    
    #param, b = sc.optimize.curve_fit(mod,[i for i in range (nb_annee*12)],data[96:nb_annee*12+96,2])
    
    plt.figure()
    plt.plot(tps,temperature)
    titre = "Precipitation à Oxford de " + str(date_deb) + " à " + str(date_deb+nb_annee)
    plt.title(titre)
    plt.xlabel("temps [annees]")
    plt.ylabel("temperature [°C]")
    plt.show()
    
    """ Le modèle testé est A*cos(omega*tps + phi) + cste """
    
    """ Initialisation des matrices pour les moindres carrées """
    
    B = temperature
    nb_data = B.shape[0]
    
    # X : vecteur des paramètres A, omega, phi, constante initialisé à 0
    X = np.array([[0,0,0,0]]).reshape(4,1)
    
    # Matrice A jacobienne du modèle linéarisé
    A = np.ones((nb_data,4))
    A[:,0] = np.cos(X[1,0]*tps + X[2,0])
    A[:,1] = -X[0,0]*tps*np.sin(X[1,0]*tps + X[2,0])
    A[:,2] = -X[0,0]*np.sin(X[1,0]*tps + X[2,0])
    
    # Matrice de poids, identité par défaut
    P = np.eye(nb_data)
    
    """ Moindres carrées """
    
    
    