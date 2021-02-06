import random
import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#ES(1 + 1)
class ES(object):
    def __init__(self, maxIter, a, sigma0, sigma, f, opseg):
        self.maxIter = maxIter        
        self.sigma0 = sigma0
        self.a = a
        self.P_S = 0
        self.sigma = sigma
        self.x = np.array([[5], [5]]) #pocetna tacka je (5,5)
        self.f = f
        self.opseg = opseg
    
    #getteri
    def getMaxIter(self):
        return self.maxIter
    
    def getA(self):
        return self.a
    
    def getP_S(self):
        return self.P_S
    
    def getSigma(self):
        return self.sigma

    def getSigma0(self):
        return self.sigma0
    
    def getX(self):
        return self.x
    
    def getEps(self):
        return self.eps
    
    #setteri
    def setMaxIter(self, maxIter):
        if (not isinstance(maxIter, int)):
            raise Exception("Maksimalan broj iteracija mora biti cijeli broj.")
        self.maxIter = maxIter
    
    def setA(self, a):
        if (not isinstance(a, float)):
            raise Exception("Parametar a mora biti float.")
        self.a = a
    
    def setP_S(self, P_S):
        if (not isinstance(P_S, float)):
            raise Exception("Parametar P_S mora biti float.")
        self.P_S = P_S
    
    def setSigma(self, sigma):
        if (not isinstance(sigma, float)):
            raise Exception("Parametar slucajnog odabira mora biti float.")
        self.sigma = sigma

    def setSigma0(self, sigma0):
        self.sigma0 = sigma0
    
    def setX(self, x):
        if (not isinstance(x, float)):
            raise Exception("Problemska varijabla mora biti float.")
        self.x = x
    
    def setEps(self, eps):
        self.eps = eps
        
    def pomnoziMatrice(self, sigma, Z):
        tmp = np.array([[0, 0]]).T
        
        for i in range(len(sigma)):
            j = 0
            while j < 1:  #zato sto je Z vektor kolona
                k = 0
                while k < 2:
                    tmp[i][j] += sigma[i][k] * Z[k][j]
                    k = k + 1
                    
                j = j + 1
                
        return tmp
        
    def mutate(self):
        #kreiramo vektor kolonu Z
        Z_0 = random.gauss(0, self.sigma)
        Z_1 = random.gauss(0, self.sigma)
        Z = np.array([[Z_0, Z_1]]).T    
        #kreiramo potomka
        potomak = self.x + self.pomnoziMatrice(self.sigma0, Z)

        #provjeravamo da li je potomak u opsegu
        #te ako nije dodjeljujemo mu bilo koju tacku unutar opsega
        if potomak[0] < self.opseg[0]:
            potomak[0] = random.uniform(self.opseg[0], self.opseg[1])
        if potomak[0] > self.opseg[1]:
            potomak[0] = random.uniform(self.opseg[0], self.opseg[1])
        if potomak[1] > self.opseg[1]:
            potomak[1] = random.uniform(self.opseg[0], self.opseg[1])
        if potomak[1] < self.opseg[0]:
            potomak[1] = random.uniform(self.opseg[0], self.opseg[1])
            
        return potomak
    
    def step(self):
        potomak = self.mutate()
        if self.f(potomak) < self.f(self.x):
            self.x = potomak
            #ovo je uspjesna mutacija
            return True
        
        return False
    
    def run(self):
        brojUspjesnihMutacija = 0
        for i in range(self.maxIter):
            if self.step():
                #ako je mutacija uspjesna
                brojUspjesnihMutacija += 1
            #azuriramo vrijednost P_S
            self.P_S = brojUspjesnihMutacija / (i + 1)
            #modificiramo sigma0 prema pravilu 1/5
            if self.P_S > 1/5:
                self.sigma0 = self.sigma0 * self.a
            else:
                if self.P_S < 1/5:
                    self.sigma0 = self.sigma0 / self.a

        return self.x

#Definisanje testnih funkcija    
def f1(x):
    return (x[0] - 3)**2 + (x[1] + 1)**2
def f2(x):
    return 20 + (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1]))
def f3(x):
    return -((1 + np.cos(12 * np.sqrt(x[0] ** 2 + x[1] ** 2))) / (0.5 * (x[0] ** 2 + x[1] ** 2) + 2))
def f4(x):
    return -(np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1 - (np.sqrt(x[0]**2+x[1]**2))/(np.pi)))))

#Graficki prikaz funkcija

#Testiranje prve funkcije
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
a = 1.3
sigma = 5
Y = f1((X1, X2))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=20.)
ax.contour(X1, X2, Y, 50, cmap='binary')
ES4 = ES(1000, a, np.array([[0.8, 0], [0, 0.8]]), sigma, f4, np.array([[-5], [5]]))
ES3 = ES(1000, a, np.array([[0.8, 0], [0, 0.8]]), sigma, f3, np.array([[-5.12], [5.12]]))
ES2 = ES(1000, a, np.array([[0.8, 0], [0, 0.8]]), sigma, f2, np.array([[-5.12], [5.12]]))
ES1 = ES(1000, a, np.array([[0.8, 0], [0, 0.8]]), sigma, f1, np.array([[-10], [10]]))

tacka = ES1.run()
print("Pronadjeni minimum za prvu funkciju:")
print(tacka)
ax.scatter(tacka[0], tacka[1], f1(tacka), color='blue', marker='x', s=150)
ax.scatter(3, -1, f1((3, -1)), color='red', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_title('$f(x_1,x_2)=(x_1-3)^2+(x_2+1)^2$')
plt.show()

#Testiranje druge funkcije
x1 = np.linspace(-5.12, 5.12, 100)
x2 = np.linspace(-5.12, 5.12, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f2((X1, X2))

plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=10.)
ax.contour(X1, X2, Y, 50, cmap='binary')

tacka = ES2.run()
print("Pronadjeni minimum za drugu funkciju:")
print(tacka)
ax.scatter(tacka[0], tacka[1], f2(tacka), color='blue', marker='x', s=150)
ax.scatter(0, 0, f2((0, 0)), color='red', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_title('$f(x_1,x_2)=20 + (x1^2 - 10cos(2\pi*x1) + x2^2 - 10cos(2\pi*x2))$')
plt.show()

#Testiranje trece funkcije
x1 = np.linspace(-5.12, 5.12, 100)
x2 = np.linspace(-5.12, 5.12, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f3((X1, X2))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=5.)
ax.contour(X1, X2, Y, 50, cmap='binary')
tacka = ES3.run()
print("Pronadjeni minimum za trecu funkciju:")
print(tacka)
ax.scatter(tacka[0], tacka[1], f3(tacka), color='blue', marker='x', s=150)
ax.scatter(0, 0, f3((0, 0)), color='red', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_title('$f(x_1,x_2)=-((1 + cos(12\sqrt{x1^2 + x2^2}))/(0.5(x1^2 + x2^2) + 2))$')
plt.show()

#Testiranje cetvrte funkcije
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f4((X1, X2))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=35.)
ax.contour(X1, X2, Y, 50, cmap='binary')
tacka = ES4.run()
print("Pronadjeni minimum za cetvrtu funkciju:")
print(tacka)
ax.scatter(tacka[0], tacka[1], f4(tacka), color='blue', marker='x', s=150)
ax.scatter(8.05502, 9.66459, f4((8.05502, 9.66459)), color='red', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_title('$f(x_1,x_2)=-|(sin(x1)cos(x2)exp(|1 - (\sqrt{x1**2+x2**2})/(\pi)|))|$')
plt.show()



    
    
        