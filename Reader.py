# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:03:58 2018

@author: Iris
"""
import numpy as np

def read(fname, date_deb, date_fin): 
    data = np.genfromtxt(fname, skip_header=7,skip_footer=13)
    return data
