# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:03:58 2018

@author: Iris
"""
import numpy as np

def read(fname, date_deb, nb_annee): 
    data = np.genfromtxt(fname, skip_header=7,skip_footer=13)
    id_deb = int((date_deb-data[0,0])*12)
    # 12*nb_annee car 12 mois par an
    temperature = data[id_deb:id_deb + 12*nb_annee,2]
    temperature = temperature.reshape(12*nb_annee,1)
    #On récupère aussi la date correspondant avec des années décimales: 1900+2/12 = mars 19000
    tps = data[id_deb:id_deb + 12*nb_annee,0] + (data[id_deb:id_deb + 12*nb_annee,1]-1)/12
    #tps = tps.reshape(12*nb_annee,1)
    return temperature, tps
