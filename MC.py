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

"""Observations"""

z = np.array([-10.171,-8.314,-6.452,-4.175,-1.996,0.179,2.816,4.170,6.734,7.356,10.894,12.254]).reshape((12,1))
y = np.array([350.826,218.434,146.850,71.788,23.079,14.813,50.694,95.434,185.819,265.981,403.319,579.559]).reshape((12,1))

l = np.zeros((24,1))

for i in range (0,12):
    l[i*2] = float(z[i])
    l[i*2+1] = float(y[i])
    
"""Pondérations"""
sigmaz = 0.4
sigmay = 0.6
sigma0 = 1
Kl = np.identity(24)
for i in range (0,12):
    Kl[i*2][i*2] = sigmaz**2
    Kl[i*2+1][i*2+1] = sigmay**2
    
    
Ql = Kl/sigma0**2
P = np.linalg.inv(Ql)



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





"""POLYNOME DE DEGRE 3"""

"""Valeurs initiales"""
#On prends a0, b0, c0 valant a, b et c trouvé avec l'approximation par un polynome de degré 2
a03 = xchap[0][0]
b03 = xchap[1][0]
c03 = xchap[2][0]
d03 = 1
x03 = np.array([a03,b03,c03,d03]).reshape((4,1))


"""itérations"""
#5 itérations pour être sûr que ça a convergé
for k in range(1):
    w3=np.zeros((12,1))
    A3=np.ones((12,4))
    G3=np.zeros((12,24))
    for i in range(len(w3)):
        w3[i][0] = x0[0]*y[i][0]**3+x0[1][0]*y[i][0]**2+x0[2][0]*y[i][0]+x0[3][0]-z[i][0]
        A3[i][0] = y[i][0]**3
        A3[i][1] = y[i][0]**2
        A3[i][2] = y[i][0]
        G3[i][i*2] = -1
        G3[i][i*2+1] = 3*x0[0][0]*y[i][0]**2+2*x0[1][0]*y[i][0]+x0[2][0]
    
    
    TMP3 = np.linalg.inv(np.dot(np.dot(G3,Ql),np.transpose(G3)))
    dxchap3 = -np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(A3),TMP3),A3)),np.transpose(A3)),TMP3),w3)
    
    xchap3 = x03+dxchap3

    S3 = np.dot(np.dot(Ql,np.transpose(G3)),TMP3)
    vchap3 = np.dot(S3,w3+np.dot(A3,dxchap3))
    lchap3 = l-vchap3
    Qxchap3 = np.linalg.inv(np.dot(np.dot(np.transpose(A3),TMP3),A3))
    Qvchap3 = np.dot(np.dot(S3,G3),Ql)-np.dot(S3,np.dot(A3,np.dot(Qxchap3,np.dot(np.transpose(A3),np.transpose(S)))))
    Qlchap3 = Ql-Qvchap3
    
    x03 = xchap3
    
    
    
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