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

def mod(t, A, omega,phi,cst):
    return A*np.cos(omega*t + phi)+ cst
    
def triang(t, A, omega,phi,B, omega2,phi2,cst):
    return B*(np.cos(omega2*t+phi2))**2 + A*np.cos(omega*t + phi)+ cst
    
def mod2(t, A, omega, B, omega2,cst):
    return A*np.sin(omega*t)+ B*np.cos(omega2*t) + cst
if __name__=="__main__":
    
    data = read("oxforddata.txt",1,1)
    nb_data = len(data)
    nb_annee =4
    
    
    param, b = sc.optimize.curve_fit(mod3,[i for i in range (nb_annee*12)],data[96:nb_annee*12+96,2])
    
    plt.figure()
    plt.plot([i for i in range (nb_annee*12)],data[0:nb_annee*12,2])
    #plt.plot([i for i in range (nb_annee*12)],[mod2(i,param[0],param[1],param[2],param[3],param[4]) for i in range (nb_annee*12)])
    #plt.plot([i for i in range (nb_annee*12)],[mod(i,param[0],param[1],param[2],param[3]) for i in range (nb_annee*12)])
    plt.plot([i for i in range (nb_annee*12)],[mod3(i,param[0],param[1],param[2],param[3],param[4],param[5],param[6]) for i in range (nb_annee*12)])

    plt.show()
    
    
    
#    gmod = lmfit.Model(mod)
#    resultAN = gmod.fit(data[0:nb_annee*12,2], x = [i for i in range (nb_annee*12)], A1 = 0, B1 = 0)
#    
#    print(resultAN.fit_report())
    
    
#    annee_debut = int(pluvio[0][0])
#    annee_fin = int(pluvio[-1][0])
#    nb_annee = annee_fin - annee_debut + 1
#    
#    rain = []
#    year = []
#    for m in range(nb_annee*12):
#        rain.append(pluvio[m][5])
#        year.append(pluvio[m][0])
#        
#    plt.title("Precipitation à Oxford de 1853 a 2016")
#    plt.xlabel("temps [annees]")
#    plt.ylabel("precipitation [mm]")
#    plt.plot(rain,"-.")
#    plt.show()