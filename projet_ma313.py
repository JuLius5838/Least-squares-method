import numpy as np 
import projet_fichier as pf
import matplotlib.pyplot as plt

######################### PARTIE 1 ##########################

#Question 1.1

def ResolMCEN(A, b):
    #L.LT.x = b     LT.x = L_inv.b
    M = np.dot(A.T,A)
    Z = np.dot(A.T,b)
    L = pf.Cholesky(M)
    Y = pf.ResolTriInf(L,Z)
    LT = np.transpose(L)
    x = pf.ResolTriSup(LT, Y)
    W = np.dot(A,x)
    Error = (np.linalg.norm(W-b))**2
    return x, Error

#Question 1.2.1

def Ker(A): 
    #Application théorème du rang et étude du noyau 

    n,p =A.shape
    rank = np.linalg.matrix_rank(A)

    if rank == p:
        kerA = 0 
        print('Ker(A) = 0')
        return kerA

    else:
        kerA = 1
        print('Ker(A) != 0')
        return kerA

def reduce_GSDecomposition(A):
    """ Calcul la décomposition QR réduite avec étude du noyau """
    
    kerA = Ker(A)
    if kerA != 0:
        print('Ker(A) != 0')
        print('impossible de réaliser QR')
        return

    n,p=A.shape

    if n > p :

        Q=np.zeros((n,p))
        R=np.zeros((p,p))
        for j in range(p):
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
    
    elif n == p :
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

    else :
        print('impossible')

#Question 1.2.2

def Resol_MCQR(A,b):
    Q,R = reduce_GSDecomposition(A)
    W = np.dot(Q.T,b)
    X = pf.ResolTriSup(R,W)
    Error = (np.linalg.norm(A@X - b))**2
    return X, Error

#Question 1.3

def ResolMCNP(A, b):
    X, res, r, s = np.linalg.lstsq(A, b, rcond=None)
    return X, res


######################### PARTIE 2 ##########################

# Matrix 1:

A1 = np.array([[1,2],[2,3],[-1,2]])
b1 = np.array([[12],[17],[6]])

# Matrix 2:

A2 = np.array([[1,21],[-1,-5],[1,17],[1,17]])
b2 = np.array([[3],[-1],[1],[1]])

#Matrix 3:

def Matrix3():
    xi = np.array([0.3,-2.7,-1.9,1.2,-2.6,2.7,2,-1.6,-0.5,-2.4])
    yi = np.array([[2.8],[-9.4],[-4.5],[3.8],[-8],[3],[3.9],[-3.5],[1.3],[-7.6]])
    Matrix = np.ones((10,3))
    Matrix[:,1]=xi 
    x2 = np.power(xi,2)
    Matrix[:,2]=x2
    b = yi
    return Matrix, b

A3,b3 = Matrix3()

#Question 2.1

#Cholesky

print('Avec Cholesky')

Xc1, ErrorC1 = ResolMCEN(A1,b1)
Xc2, ErrorC2 = ResolMCEN(A2,b2)
Xc3, ErrorC3 = ResolMCEN(A3,b3)

print('X1 = ', Xc1, '\n ; ||AX - b||^2 = ', ErrorC1)
print('X2 = ', Xc2, '\n; ||AX - b||^2 = ', ErrorC2)
print('X3 = ', Xc3, '\n; ||AX - b||^2 = ', ErrorC3)


#QR

print('Avec QR')

Xqr1, ErrorQR1 = Resol_MCQR(A1,b1)
Xqr2, ErrorQR2 = Resol_MCQR(A2,b2)
Xqr3, ErrorQR3 = Resol_MCQR(A3,b3)

print('X1 = ', Xqr1, '\n ; ||AX - b||^2 = ', ErrorQR1)
print('X2 = ', Xqr2, '\n; ||AX - b||^2 = ', ErrorQR2)
print('X3 = ', Xqr3, '\n; ||AX - b||^2 = ', ErrorQR3)

#Avec fonction numpy

print('Avec numpy')

Xnp1, ErrorNP1 = ResolMCNP(A1,b1)
Xnp2, ErrorNP2 = ResolMCNP(A2,b2)
Xnp3, ErrorNP3 = ResolMCNP(A3,b3)

print('X1 = ', Xnp1, '\n ; ||AX - b||^2 = ', ErrorNP1)
print('X2 = ', Xnp2, '\n; ||AX - b||^2 = ', ErrorNP2)
print('X3 = ', Xnp3, '\n; ||AX - b||^2 = ', ErrorNP3)


#Question 2.2


def Checking(A,b):
    xstar,estar = ResolMCNP(A,b)
    n,m = xstar.shape
    xstar*10**(-3)
    Nexp = 10**6
    for i in range(Nexp):
        x = (np.random.rand(n,m))*10**(-3)
        difx = np.linalg.norm(x - xstar)
        xstarnorm = np.linalg.norm(xstar)
        dif = xstarnorm - difx
        print(dif)
        if dif < 10**(-3):
            Solve = (np.linalg.norm(A@x - b))**2
            if Solve >= estar:
                print('minimal')
            else:
                print('not good')
                break


Checking(A1,b1)

def matrice(x,y):
    n = np.size(x)
    A = np.zeros((n,3))
    b = np.zeros((n,1))

    for i in range(0,n):
        A[i,0] = 2*x[i]
        A[i,1] = 2*y[i]
        A[i,2] = 1
        b[i,0] = x[i]**2 + y[i]**2

    return A, b


x, y = pf.donnees_partie3()
A, b = matrice(x, y)
X1, error = ResolMCNP(A, b)

print('||Ax - b||² = ', error)
print('x = ', X1)

alpha = X1[0]
beta = X1[1]
gamma = X1[2]
radius = (gamma + alpha**2 + beta**2)**(1/2)

print('alpha = ', alpha)
print('beta = ', beta)
print('gamma = ', gamma)
print('radius = ', radius)


plot, axes = plt.subplots()

plt.plot(x, y, 'og')
plt.plot(alpha, beta, 'ob')
plt.axis([-6, 8, -8, 7])
plt.grid()

circle = plt.Circle((alpha, beta), radius, color='green', fill=False)
axes.add_artist(circle)
plt.show()






