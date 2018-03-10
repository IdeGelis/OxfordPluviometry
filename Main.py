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

def mod(tps, param):
    """
    Modèle A*cos(omega*tps + phi) + cste
    """ 
    res = param[0,0]*np.cos(param[1,0]*tps + param[2,0])+ param[3,0]
    return res
    

    
if __name__=="__main__":
    nb_annee = 50
    # Date du début de l'étude, date_deb >=1853
    date_deb = 1853   
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
    nb_data = temperature.shape[0]
    
    # X : vecteur des paramètres A, omega, phi, constante initialisé à 0
    X = np.array([[20,(2*np.pi),np.pi,13.69]]).reshape(4,1)
    
    # Données
    l = temperature
    
    # Matrice de poids, identité par défaut
    Ql = np.eye(nb_data)
    P = np.eye(nb_data)
    
    """ Moindres carrées """
    
    # Itérations 
    for k in range(10):
        # Matrice A jacobienne du modèle linéarisé
        A = np.ones((nb_data,4))
        A[:,0] = np.cos(X[1,0]*tps + X[2,0])
        A[:,1] = -X[0,0]*tps*np.sin(X[1,0]*tps + X[2,0])
        A[:,2] = -X[0,0]*np.sin(X[1,0]*tps + X[2,0])
        
        B = l - mod(tps,X).reshape((nb_data,1))
        
        # N = A.T*P*A, K = A.T*P*B
        N = np.dot(np.dot(A.T,P),A)
        K = np.dot(np.dot(A.T,P),B)
        
        # dXchap = inv(N)*K
        dXchap = np.dot(np.linalg.inv(N), K)
        
        Xchap = X+dXchap

        vchap = B - np.dot(A,dXchap)
        lchap = l-vchap
        
        # Matrice de variance covariance
        Qxchap = np.linalg.inv(np.dot(np.dot(np.transpose(A),P),A))
        Qvchap = Ql - np.dot(np.dot(A,Qxchap),A.T)
        Qlchap = Ql-Qvchap
        
        sigma0_2 = np.dot(np.dot(vchap.T,P),vchap)/(nb_data - 4)
        print(sigma0_2)
        
        X = Xchap
    
    """ Affichage des résultats des MC """
    
    plt.figure()
    plt.plot(tps,temperature, "o", label = "Observations")
    plt.plot(tps, mod(tps,X))
    #Test pour les params initiaux
    #•plt.plot(tps, mod(tps,np.array([[20,(2*np.pi),np.pi,13.69]]).reshape(4,1)))
    titre = "Precipitation à Oxford de " + str(date_deb) + " à " + str(date_deb+nb_annee)
    plt.title(titre)
    plt.xlabel("temps [annees]")
    plt.ylabel("temperature [°C]")
    plt.show()
    
    """
    Les MC sont fait et il semble fonctionner cependant seulement quand on donne
    des valeur initiales pas trop éloignées de la véritées.
    Il faut maintenant faire les graphes de résidus, etc...
    Par contre sigma0_2 semble se stabiliser très vite mais proche de 2.6 et non de 1...???
    """
    
        
    


    
    
    