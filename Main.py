# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:09:59 2018

@author: Iris and Mannaig
"""

import numpy as np
import matplotlib.pyplot as plt


pluvio = np.genfromtxt("oxforddata.txt",skip_header=7,skip_footer=13)

annee_debut = int(pluvio[0][0])
annee_fin = int(pluvio[-1][0])
nb_annee = annee_fin - annee_debut + 1

rain = []
year = []
for m in range(nb_annee*12):
    rain.append(pluvio[m][5])
    year.append(pluvio[m][0])
    
plt.title("Precipitation à Oxford de 1853 a 2016")
plt.xlabel("temps [annees]")
plt.ylabel("precipitation [mm]")
plt.plot(rain,"-.")
plt.show()