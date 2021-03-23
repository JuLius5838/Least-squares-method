# -*- coding: utf-8 -*-



import numpy as np


def DecompositionGS(A):
    """ Calcul de la décomposition QR de A une matrice carrée.
    L'algorithme de Gram-Schmidt est utilisé.
    La fonction renvoit (Q,R) """
    n,m=A.shape
    if n !=m :
        raise Exception('Matrice non carrée')

    Q=np.zeros((n,n))
    R=np.zeros((n,n))
    for j in range(n):
        for i in range(j):
            R[i,j]=Q[:,i]@A[:,j]
        w=A[:,j]
        for k in range(j):
            w=w-R[k,j]*Q[:,k]
        norme=np.linalg.norm(w)
        if norme ==0:
            raise Exception('Matrice non inversible')
        R[j,j]=norme
        Q[:,j]=w/norme
    return Q,R

def ResolTriSup(T,b):
    """Résolution d'un système triangulaire supérieur carré
    Tx=b
    La fonction ne vérifie pas la cohérence des dimensions de T et b
    ni que T est triangulaire supérieure.
    La fonction rend x sous la forme du même format que b."""


    n,m=T.shape
    x=np.zeros(n)
    for i in range(n-1,-1,-1):
        S=T[i,i+1:]@x[i+1:]
        x[i]=(b[i]-S)/T[i,i]
    x=np.reshape(x,b.shape)
    return x

def ResolTriInf(T,b):
    """Résolution d'un système triangulaire inférieur carré
    Tx=b
    La fonction ne vérifie pas la cohérence des dimensions de T et b
    ni que T est triangulaire inférieure.
    La fonction rend x sous la forme du même format que b."""


    n,m=T.shape
    x=np.zeros(n)
    for i in range(n):
        S=T[i,:i]@x[:i]
        x[i]=(b[i]-S)/T[i,i]
    x=np.reshape(x,b.shape)
    return x

def Cholesky(A):
    """
    Fonction qui calcule L la matrice de la décomposition de
    Cholesky de A une matrice réelle symétrique définie positive
    (A=LL^T où L est triangulaire inférieure).
    La fonction ne vérifie pas que A est symétrique.
    La fonction rend L.
    """

    n,m=A.shape
    if n != m:
        raise Exception('Matrice non carrée')
    L=np.zeros((n,n))
    for i in range(n):
        s=0.
        for j in range(i):
            s=s+L[i,j]**2
        R=A[i,i]-s
        if R<=0:
            raise Exception('Matrice non définie positive')
        L[i,i]=np.sqrt(R)
        for j in range(i+1,n):
            s=0.
            for k in range(i):
                s=s+L[i,k]*L[j,k]
            L[j,i]=(A[j,i]-s)/L[i,i]
    return L


def donnees_partie3():
    """ Fonction qui donne les données à traiter dans la partie 3
    du projet. 
    
    ----------------
    Utilisation : x,y=donnes_partie3()
    """
    
    x=np.array([ 3.58, -2.26,  1.17,  7.09,  1.3 , -4.82, -4.83,  1.53,  5.73,
       -3.44,  4.04,  2.99,  3.59, -4.66, -0.61,  0.67, -4.02, -1.91,
        6.58, -5.07,  5.18, -1.67, -2.6 ,  4.27,  4.  , -5.36, -2.1 ,
        5.94, -3.92, -3.29,  6.39,  2.04, -4.66,  7.73, -4.26,  4.26,
       -4.15, -4.67, -0.73, -4.8 ,  5.15, -2.9 ,  6.55,  5.7 ,  6.15,
        5.46,  0.1 , -2.46, -4.52,  7.01,  6.79, -0.04,  7.25, -2.01,
        7.07, -2.02, -4.57,  3.11,  1.01,  6.38, -4.69,  7.19, -4.22,
       -5.08,  6.9 ,  4.28, -3.31,  6.58, -1.71,  6.28, -3.9 ,  6.88,
       -3.76,  4.53,  6.31,  6.54,  7.17,  7.3 ,  6.38, -1.17, -0.22,
       -0.64,  3.91,  2.11,  1.66, -1.66, -4.1 ,  6.16,  7.54, -1.44,
        5.57,  4.85,  7.04, -4.64,  6.67, -4.93,  6.92, -3.11,  0.17,
        3.95])

    y=np.array([-6.71,  4.17, -6.94,  0.19, -6.9 , -2.03, -0.52,  5.14, -4.87,
       -4.94,  4.12,  4.99,  4.83,  0.78, -6.75, -7.3 , -4.05,  4.58,
       -3.26, -0.12,  4.38,  4.14, -5.99,  4.38, -6.41,  0.03, -6.2 ,
        3.04,  2.  ,  3.26,  2.53,  4.94, -2.35,  0.85, -4.18,  4.79,
        1.99, -2.74, -6.75, -2.97,  3.87, -5.26, -4.45,  3.87,  2.74,
       -5.24, -6.97,  4.07, -2.31, -0.11,  1.71, -7.01, -0.62, -5.79,
        0.82, -5.86,  1.87,  5.09, -7.09, -3.97, -2.9 , -2.3 , -4.63,
       -1.17,  0.86,  4.68,  3.23, -3.48, -6.54, -4.76,  2.55,  0.92,
        3.19,  3.97,  2.93, -3.28,  0.26, -2.52, -4.63, -6.4 ,  5.05,
       -6.75, -6.41,  5.18,  5.49,  4.13,  2.57,  2.58,  0.04,  4.8 ,
        3.59,  4.09,  1.15, -2.68,  1.24, -0.9 , -3.56,  3.29, -7.06,
        4.46])

    return x,y


# def Matrix_A(size):         # créer une matrice A définie positive
#     M = np.random.randint(-10, +10, size = (size, size))
#     A = np.dot(M.T, M)
#     return A

# def Matrix_B(size):         # créer une matrice colonne B de la même taille que A
#     B = np.random.randint(-10, 10, size = (size, 1))
#     return B