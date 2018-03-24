# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:09:59 2018

@author: Iris and Mannaig
"""

import numpy as np
import matplotlib.pyplot as plt
from Reader import read

import scipy as sc
#import lmfit

def mod(tps, param):
    """
    Modèle A*cos(omega*tps + phi) + cste
    """ 
    res = param[0,0]*np.cos(param[1,0]*tps + param[2,0])+ param[3,0]
    return res
    
def MC(temperature,tps):
    """
    Moindres carrés
    """
    # X : vecteur des paramètres A, omega, phi, constante initialisé à 0
    X = np.array([[20,(2*np.pi),np.pi,13.69]]).reshape(4,1)

    # Données
    l = temperature
    nb_data = np.size(l)
    print(nb_data)
    
    # Matrice de poids, identité par défaut
    Ql = np.eye(nb_data)
    P = np.eye(nb_data)
    
    """ Moindres carrées """
    
    # Itérations 
    for k in range(10):
        # Matrice A jacobienne du modèle linéarisé
        A = np.ones((nb_data,4))
        A[:,0] = np.cos(X[1,0]*tps[:,0] + X[2,0])
        A[:,1] = -X[0,0]*tps[:,0]*np.sin(X[1,0]*tps[:,0] + X[2,0])
        A[:,2] = -X[0,0]*np.sin(X[1,0]*tps[:,0] + X[2,0])
        
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
        
        
    plt.figure()
    plt.title("Histogramme des résidus")
    # Afficher la courbe de la loi normale de moyenne 0 et d'écart type sigma0
    #x = np.linspace(0, 6*np.sqrt(sigma0_2), 100)
    #plt.plot(x,sc.stats.norm.pdf(x,0,np.sqrt(sigma0_2)))
    plt.hist(vchap)
    plt.show()
    
    print ('Sigma0_2 :', sigma0_2)
    
    return (X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap)
    
if __name__=="__main__":
    nb_annee = 10
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
    
    """ 
    Le modèle testé est A*cos(omega*tps + phi) + cste 
    Il y a donc 4 paramètres.    
    """
    
    """ Initialisation des matrices pour les moindres carrées """
    nb_data = temperature.shape[0]
    
    X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap = MC(temperature,tps)
    """
    Les MC sont fait et il semble fonctionner cependant seulement quand on donne
    des valeur initiales pas trop éloignées de la véritées.
    Il faut maintenant faire les graphes que David voulait pour une estimation par MC la plus complète possible.
    Par contre sigma0_2 semble se stabiliser très vite mais proche de 2.6 et non de 1...???

    Rajouter matrice de variance co-variance!!!
    
    """ 
        
    """ Faire élimination des points faux en mode automatique """
    
    """
    Poser un critere d'élimination : pex à 3*sigma pour trouver et éliminer les erreurs dans les données
    ou
    Estimer de manière itérative des droites de regression en eliminant a chaque iteration le 
    residu le plus fort. Poser un critere d'arret=0.8 pex
    
    Idee : Calculer l'écart entre la valeur et le modele estime. Puis faire un predicat sur cette valeur :
        si |ecart| > ... : elimination
        si ecart <= ... : conservation
    Puis vérification du modèle.
    """
    
    
    
    """ RANSAC """
    """
    Objectif: Ajusteement robuste d'un modèle à un jeu de données S contenant des points faux.
    """
    
    t = 2
    T = 3*nb_data/4
    K = 10
    # La représentation discrète d'un signal exige des échantillons régulièrement espacés à une fréquence d'échantillonnage supérieure au double de la fréquence maximale présente dans ce signal.
    # Frequence = 2*pi/T
    # T = 12 mois
    # Frequence  =  2*pi/12 = pi/6
    # Th de Shanon Freq_echantillonage > 2*pi/6 = pi/3 
    # Ce qui implique un point tous les 6 mois
    
    nb_mois = nb_data # Une donnée par mois!
    # n nb minial de données pour estimer le modèle: selon le théorème de Shanon
    n = nb_mois//6
    jd_temperature = np.zeros((n,1))
    jd_tps = np.zeros((n,1))
        
    ite = 0    
    
    meilleur_ens_pts_temperature = []
    meilleur_ens_pts_tps = []
    
    sz_max_ens = 0
    
    while ite < K : 
        ens_pts_temperature = []
        ens_pts_tps = []
        
        # Tirage des valeurs initiales
        tmp = 0       
        for i in range (n):
            aleat = np.random.randint(6)
            jd_temperature[i,0] = temperature[tmp+aleat]
            jd_tps[i,0] = tps[tmp+aleat]
            tmp += 6
        
        # Graphe des points aléatoires choisi
        plt.figure()
        plt.plot(jd_tps,jd_temperature, "o", label = "Observations sélectionnées au tirage aléatoire")
        titre = "Precipitation à Oxford de " + str(date_deb) + " à " + str(date_deb+nb_annee)
        plt.title(titre)
        plt.xlabel("temps [annees]")
        plt.ylabel("temperature [°C]")
        plt.show()
        
        X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap = MC(jd_temperature,jd_tps)
        
        # Selection des points qui collent au modèle
        for i in range (nb_data):
            if np.abs(mod(tps[i,0],X) - temperature[i,0]) < t:
               ens_pts_temperature.append(temperature[i,0])
               ens_pts_tps.append(tps[i,0])
        
        print(len(ens_pts_temperature))
        if len(ens_pts_temperature)>=sz_max_ens:
            meilleur_ens_pts_temperature = ens_pts_temperature
            meilleur_ens_pts_tps = ens_pts_tps
               
        # Si suffisement de points collent au modèle, la boucle est arrêtée
        if (len(ens_pts_temperature)>=T):         
            sz_max_ens = len(ens_pts_temperature)
            arr_ens_pts_temperature = np.reshape(ens_pts_temperature,(len(ens_pts_temperature),1))
            arr_ens_pts_tps = np.reshape(ens_pts_tps,(len(ens_pts_tps),1))
    
            X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap = MC(arr_ens_pts_temperature, arr_ens_pts_tps)
            
            break
        print(ite)
        ite+=1
        
    
    # si on n'est pas sortie de la boucle du fait qu'on a trouvé un ensemble avec beaucoup de point
    if ite >K:
        arr_ens_pts_temperature = np.reshape(meilleur_ens_pts_temperature,(len(meilleur_ens_pts_temperature),1))
        arr_ens_pts_tps = np.reshape(meilleur_ens_pts_tps,(len(meilleur_ens_pts_tps),1))
        
        X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap = MC(arr_ens_pts_temperature, arr_ens_pts_tps)
 
        
                
        


    
    
    