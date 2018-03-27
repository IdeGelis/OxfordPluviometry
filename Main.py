# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:09:59 2018

@author: Iris and Mannaig
"""

import numpy as np
import matplotlib.pyplot as plt
from Reader import read
from matplotlib import cm as cm
import scipy as sc
#import lmfit



def mod(tps, param):
    """
    Modèle A*cos(omega*tps + phi) + cste
    """ 
    res = param[0,0]*np.cos(param[1,0]*tps + param[2,0]) + param[3,0]
    return res




def MC(temperature,tps,mode):
    """
    Moindres carrés
    """
    
    # X : vecteur des paramètres A, omega, phi, constante initialisé 
    X = np.array([[7.5,(2*np.pi),0,14]]).reshape(4,1)

    # Données
    l = temperature
    nb_data = np.size(l)
    
    # Matrice de poids : identité par défaut
    Ql = np.eye(nb_data)
    P = np.linalg.inv(Ql)
    
    # Valeurs arbitraires pour rentrer dans la boucle itérative
    sigma0_2_last = 10000.0
    sigma0_2 = 13000.0
    
    # Itérations 
    while sigma0_2 != sigma0_2_last :
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
        dXchap = np.dot(np.linalg.inv(N),K)
        
        Xchap = X + dXchap

        # Résidus
        vchap = B - np.dot(A,dXchap)
        lchap = l - vchap
        
        # Matrice de variance covariance
        Qxchap = np.linalg.inv(np.dot(np.dot(np.transpose(A),P),A))
        Qvchap = Ql - np.dot(np.dot(A,Qxchap),A.T)
        Qlchap = Ql - Qvchap
        
        # Résidus normalisés
        vnorm = np.divide(vchap,(np.sqrt(np.diag(Qvchap))).reshape(1800,1))

        sigma0_2_last = sigma0_2
        sigma0_2 = np.dot(np.dot(vchap.T,P),vchap)/(nb_data - 4)

        X = Xchap
        
        if mode == "elim" :
            P = PtsFaux2(P,vchap)

    
    return (X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap, vchap,vnorm)




def PtsFaux2(poids,residus):
    """
    Elimination des points faux
    Modification du poids de la valeur ayant le résidu le plus élevé
    """
    ir1 = np.argmax(residus)
    poids[ir1][ir1] = 0.0
    return poids




def testSeuil(residus, seuil):
    cpt = 0
    for i in range (residus.shape[0]):
        if residus[i,0] <= seuil : 
            cpt +=1
    return cpt*100/residus.shape[0]


def Gauss(x,mu,sigma):
    """
    Création d'une gaussienne (es fonctions pré'implémentées ne marchent pas...)
    """
    coeff = 1/(sigma*np.sqrt(2*np.pi))
    gauss = 400+ coeff * np.exp(-1/2*((x-mu)/sigma)**2)
    return gauss


def ransac(t,T,K,temperature, tps):
    """ RANSAC """
    """
    Objectif: Ajustement robuste d'un modèle à un jeu de données S contenant des points faux.
    """
    
    nb_mois = tps.shape[0] # Une donnée par mois!
    # n nb minial de données pour estimer le modèle: selon le théorème de Shanon
    n = nb_mois//6
    jd_temperature = np.zeros((n,1))
    jd_tps = np.zeros((n,1))
        
    ite = 0    
    
    meilleur_ens_pts_temperature = []
    meilleur_ens_pts_tps = []
    
    sz_max_ens = 0
    
    while ite < K :
        #print("Iteration n° ",ite)
        ens_pts_temperature = []
        ens_pts_tps = []
        
        # La représentation discrète d'un signal exige des échantillons régulièrement espacés à une fréquence d'échantillonnage supérieure au double de la fréquence maximale présente dans ce signal.
        # Frequence = 1/T
        # T = 12 mois
        # Th de Shanon Freq_echantillonage > 2/12 = 1/6
        # Ce qui implique un point tous les 6 mois
        
        # Tirage des valeurs initiales
        tmp = 0       
        for i in range (n):
            aleat = np.random.randint(6)
            jd_temperature[i,0] = temperature[tmp+aleat]
            jd_tps[i,0] = tps[tmp+aleat]
            tmp += 6
        
        # Graphe des points aléatoires choisi
#        plt.figure()
#        plt.plot(jd_tps,jd_temperature, "o", label = "Observations sélectionnées au tirage aléatoire")
#        titre = "Precipitation à Oxford de " + str(date_deb) + " à " + str(date_deb+nb_annee)
#        plt.title(titre)
#        plt.xlabel("temps [annees]")
#        plt.ylabel("temperature [°C]")
#        plt.show()
        
        X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap, vchap, vnorm = MC(jd_temperature,jd_tps,mode="notElim")
        
        # Selection des points qui collent au modèle
        for i in range (nb_data):
#            print(tps[i,0], ' : ', np.abs(mod(tps[i,0],X) - temperature[i,0]))
            if np.abs(mod(tps[i,0],X) - temperature[i,0]) < t:
               ens_pts_temperature.append(temperature[i,0])
               ens_pts_tps.append(tps[i,0])
        
        #print("Taille de l'ensemble de points collant au modèle :", len(ens_pts_temperature))
        if len(ens_pts_temperature)>=sz_max_ens:           
            meilleur_ens_pts_temperature = ens_pts_temperature
            meilleur_ens_pts_tps = ens_pts_tps
            sz_max_ens = len(meilleur_ens_pts_temperature)
               
        # Si suffisement de points collent au modèle, la boucle est arrêtée
        if (len(ens_pts_temperature)>=T):         
            sz_max_ens = len(ens_pts_temperature)
            arr_ens_pts_temperature = np.reshape(meilleur_ens_pts_temperature,(len(meilleur_ens_pts_temperature),1))
            arr_ens_pts_tps = np.reshape(meilleur_ens_pts_tps,(len(meilleur_ens_pts_tps),1))
    
            X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap, vchap, vnorm = MC(arr_ens_pts_temperature, arr_ens_pts_tps, mode="notElim")
            
            break
        
        ite+=1
        
    
    # si on n'est pas sortie de la boucle du fait qu'on a trouvé un ensemble avec beaucoup de point
    if ite >= K:
        arr_ens_pts_temperature = np.reshape(meilleur_ens_pts_temperature,(len(meilleur_ens_pts_temperature),1))
        arr_ens_pts_tps = np.reshape(meilleur_ens_pts_tps,(len(meilleur_ens_pts_tps),1))
        
        X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap, vchap, vnorm = MC(arr_ens_pts_temperature, arr_ens_pts_tps, mode="notElim")
    
    return (meilleur_ens_pts_temperature, meilleur_ens_pts_tps, X, sigma0_2, Qlchap, Qvchap, Qxchap, lchap, vchap)








if __name__=="__main__":
    nb_annee = 150
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
    NOTE DE DAVID!:
    Il faut divisé les résidus par leur écarts type (donc racine de la diagonale de Qvchap)
    Division avant le test d'elimination des points faux mais aussi pour tracer les histogrammes
	
	
    residus divisé par leurs ecarts-types 99.9% entre -3 3
	
	 RANSAC: regarder le pourcentage de points eliminer par la méthode des points faux pour eliminer un pourcentage correspondant par Ransac
	 Ajouter dans le rapport que c'est subjectif ce choix de paramètres
	
	
	 PEnser au Chi-2
	
	 Penser au aux courbes des normales 
    """
    
    
    """
    Fait : résidus normalisés ; imprimer les histogrammes des résidus normalisés ; mode d'itération : arrêt quand sigma0² stagne
    Reste à faire : courbe normale ; elimination pts faux MC ; residus divisé par leurs ecarts-types 99.9% entre -3 3 ; chi²
    """
    
    
    
    """ 
    Le modèle testé est A*cos(omega*tps + phi) + cste 
    Il y a donc 4 paramètres.    
    """
    
    """ Initialisation des matrices pour les moindres carrées """
    nb_data = temperature.shape[0]
    
    print("---------------------------MOINDRES-CARRES-----------------------------------")
    # Moindres-carré
    X_MC, sigma0_2_MC, Qlchap_MC, Qvchap_MC, Qxchap_MC, lchap_MC, vchap_MC, vnorm_MC = MC(temperature,tps,mode="notElim")
    
    """ Affichage des résultats des MC """

    plt.figure()
    plt.plot(tps,temperature, label = "Observations")
    plt.plot(tps, mod(tps,X_MC), label = "Modèle issus des MC")
    #Test pour les params initiaux
    #•plt.plot(tps, mod(tps,np.array([[20,(2*np.pi),np.pi,13.69]]).reshape(4,1)))
    titre = "Precipitation à Oxford de " + str(date_deb) + " à " + str(date_deb+nb_annee)
    plt.title(titre)
    plt.xlabel("temps [annees]")
    plt.ylabel("temperature [°C]")
    plt.legend()
    plt.show()
        
    plt.figure()
    plt.plot(tps[50:50+20*12,:],temperature[50:50+20*12,:], label = "Observations")
    plt.plot(tps[50:50+20*12,:], mod(tps[50:50+20*12,:],X_MC), label = "Modèle issus des MC")
    #Test pour les params initiaux
    #•plt.plot(tps, mod(tps,np.array([[20,(2*np.pi),np.pi,13.69]]).reshape(4,1)))
    titre = "Zoom sur les precipitations à Oxford de " + str(date_deb+50) + " à " + str(date_deb+70)
    plt.title(titre)
    plt.xlabel("temps [annees]")
    plt.ylabel("temperature [°C]")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('fonction gaussienne')
    x = np.linspace(0, 6*np.sqrt(sigma0_2_MC), 100)
    y = Gauss(x,0,np.sqrt(sigma0_2_MC))
    plt.plot(x,y)
    plt.show()
    
    plt.figure()
    plt.title("Histogramme des résidus")
    # Afficher la courbe de la loi normale de moyenne 0 et d'écart type sigma0

#    plt.plot(x,sc.stats.norm.pdf(x,0,np.sqrt(sigma0_2_MC)))
    plt.hist(vchap_MC)
    plt.show()
    print ('Sigma0_2 :', sigma0_2_MC[0][0])
    
    plt.figure()
    plt.title("Histogramme des résidus normalisés")
    plt.hist(vnorm_MC)
    plt.show()
    print ('Sigma0_2 :', sigma0_2_MC[0][0])
    
    
    print("------------------MOINDRES-CARRES ELIMINATION PTS FAUX-----------------------")
    # Moindres-carré
    X_MC2, sigma0_2_MC2, Qlchap_MC2, Qvchap_MC2, Qxchap_MC2, lchap_MC2, vchap_MC2,vnorm_MC2 = MC(temperature,tps,mode="elim")
    
    """ Affichage des résultats des MC """

    plt.figure()
    plt.plot(tps,temperature, "o", label = "Observations")
    plt.plot(tps, mod(tps,X_MC2))
    #Test pour les params initiaux
    #•plt.plot(tps, mod(tps,np.array([[20,(2*np.pi),np.pi,13.69]]).reshape(4,1)))
    titre = "Precipitation à Oxford de " + str(date_deb) + " à " + str(date_deb+nb_annee)
    plt.title(titre)
    plt.xlabel("temps [annees]")
    plt.ylabel("temperature [°C]")
    plt.legend()
    plt.show()
        
        
    plt.figure()
    plt.title("Histogramme des résidus")
    # Afficher la courbe de la loi normale de moyenne 0 et d'écart type sigma0
    #x = np.linspace(0, 6*np.sqrt(sigma0_2), 100)
    #plt.plot(x,sc.stats.norm.pdf(x,0,np.sqrt(sigma0_2)))
    plt.hist(vchap_MC2)
    plt.show()
    print ('Sigma0_2 :', sigma0_2_MC2[0][0])
    
    plt.figure()
    plt.title("Histogramme des résidus normalisés")
    plt.hist(vnorm_MC2)
    plt.show()
    print ('Sigma0_2 :', sigma0_2_MC2[0][0])
    
    
    
    
    """
    Rajouter matrice de variance co-variance!!!
    Et le test du CHI-2
    
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
    
#    print("------------------------------RANSAC--------------------------------------")
#    
#
#    
#    t = 2 #Choisi nottament grace à testSeuil
#    T = 10*nb_data/10
#    K = 100
#    sel_temperature, sel_tps, X_ransac, sigma0_2_ransac, Qlchap_ransac, Qvchap_ransac, Qxchap_ransac, lchap_ransac, vchap_ransac = ransac(t, T, K, temperature, tps)
#
#    
#    plt.figure()
#    plt.plot(tps,temperature, "o", label = "Observations")
#    plt.plot(sel_tps,sel_temperature, "o", label = "Observations sélectionnées")
#    plt.plot(tps, mod(tps,X_ransac))
#    titre = "Precipitation à Oxford de " + str(date_deb) + " à " + str(date_deb+nb_annee)
#    plt.title(titre)
#    plt.xlabel("temps [annees]")
#    plt.ylabel("temperature [°C]")
#    plt.legend()
#    plt.show()
#        
#        
#    plt.figure()
#    plt.title("Histogramme des résidus avec la méthode ransac")
#    # Afficher la courbe de la loi normale de moyenne 0 et d'écart type sigma0
#    #x = np.linspace(0, 6*np.sqrt(sigma0_2), 100)
#    #plt.plot(x,sc.stats.norm.pdf(x,0,np.sqrt(sigma0_2)))
#    plt.hist(vchap_ransac)
#    plt.show()
#    print ('Sigma0_2 :', sigma0_2_ransac[0][0])
# 
#    print(len(sel_tps))
#    print(len(sel_tps)*100/1800, '%')
        
                
        


    
    
    