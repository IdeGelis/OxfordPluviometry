# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:03:58 2018

@author: Iris
"""
import numpy as np

def read(fname, date_deb, nb_annee): 
    data = np.genfromtxt(fname, skip_header=7,skip_footer=13)
    id_deb = (date_deb-data[0,0])*12
    print(data[id_deb,0:1])
    temperature = data[id_deb:id_deb + nb_annee,2]
    return temperature
