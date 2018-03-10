# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 10:17:13 2018

@author: Iris
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:03:35 2017

@author: Iris
"""

#Enoncé de compensation par moindres carrés, méthode de Gauss-Helmert


import numpy as np
import matplotlib.pyplot as plt


""" POLYNOME DE DEGRE 2"""

"""Valeurs initiales"""
a0 = 1
b0 = 1
c0 = 1
x0 = np.array([a0,b0,c0]).reshape((3,1))


"""itérations"""
#5 itérations pour être sûr que ça a convergé
for k in range(1):
    w=np.zeros((12,1))
    A=np.ones((12,3))
    G=np.zeros((12,24))
    for i in range(len(w)):
        w[i][0] = x0[0]*y[i][0]**2+x0[1][0]*y[i][0]+x0[2][0]-z[i][0]
        A[i][0] = y[i][0]**2
        A[i][1] = y[i][0]
        G[i][i*2] = -1
        G[i][i*2+1] = 2*x0[0][0]*y[i][0]+x0[1][0]
    
    
    TMP = np.linalg.inv(np.dot(np.dot(G,Ql),np.transpose(G)))
    dxchap = -np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(A),TMP),A)),np.transpose(A)),TMP),w)
    
    xchap = x0+dxchap

    S = np.dot(np.dot(Ql,np.transpose(G)),TMP)
    vchap = np.dot(S,w+np.dot(A,dxchap))
    lchap = l-vchap
    Qxchap = np.linalg.inv(np.dot(np.dot(np.transpose(A),TMP),A))
    Qvchap = np.dot(np.dot(S,G),Ql)-np.dot(S,np.dot(A,np.dot(Qxchap,np.dot(np.transpose(A),np.transpose(S)))))
    Qlchap = Ql-Qvchap
    
    x0 = xchap



    
"""Affichage"""
#Affichage sur graphe
X=np.arange(-15,15,0.1)
Y0=[a0*X[i]**2+b0*X[i]+c0 for i in range(len(X))]
Ychap=[xchap[0][0]*X[i]**2+xchap[1][0]*X[i]+xchap[2][0] for i in range(len(X))]
Ychap3=[xchap3[0][0]*X[i]**3+xchap3[1][0]*X[i]**2+xchap3[2][0]*X[i]+xchap3[3][0] for i in range(len(X))]

Xobs=z
Yobs=y
plt.plot(Xobs, Yobs, "o", label="Observations")

plt.plot(X,Ychap, color="r", label="Xchapeau Polynome de degre 2")
plt.plot(X,Ychap3, color="g", label="Xchapeau Polynome de degre 3")
plt.legend()
plt.show()