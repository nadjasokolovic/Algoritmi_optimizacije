#Optimizacija rojem cestica

import random as r
import copy
import numpy as np
import matplotlib.pyplot as plt

def PSO(f, opseg, N, M, eps, P):
    #inicijalizacija populacije cestica sa slucajno odabranim podacima i brzinama 
    #u problemskom prostoru
    cestice = []
    brzine = []

    for i in range(M):
        cestice.append([r.uniform(opseg[0], opseg[1]), r.uniform(opseg[0], opseg[1])])
    for i in range(M):
        brzine.append([r.uniform((opseg[0] - opseg[1]) / 2, (opseg[1] - opseg[0]) / 2), r.uniform((opseg[0] - opseg[1]) / 2, (opseg[1] - opseg[0]) / 2)])

    #parametri algoritma
    W = P[0]
    C1 = P[1]
    C2 = P[2]
    
    #pocetno rjesenje je prva cestica
    rjesenje = cestice[0]

    #sve dok se ne ispuni uslov zaustavljanja algoritma: maksimalan broj iteracija
    i = 0
    while(i < N):

        pb = cestice[0] #lokacija u dimenziji sa najboljom vrijednosti funkcije kriterija
        #prolazimo kroz sve ostale cestice da bismo pronasli najbolje pb
        for j in range(1, M):
            if f(*cestice[j]) < f(*pb):
                pb = copy.deepcopy(cestice[j])

        #drugi kriterij zaustavljanja
        if f(*pb) < f(*rjesenje):
            if abs(f(*pb) - f(*rjesenje)) < eps:
                return pb
            
            rjesenje = copy.deepcopy(pb)

        #za svaku cesticu
        for j in range(M):
            for k in range(2):

                #slučajni brojevi iz opsega [0, 1]
                r1 = r.uniform(0, 1)
                r2 = r.uniform(0, 1)

                #racunanje nove brzine 
                brzine[j][k] = W[k] * brzine[j][k] + C1[k] * r1 * (pb[k] - cestice[j][k]) + C2[k] * r2 * (rjesenje[k] - cestice[j][k])

                #racunanje nove pozicije cestice
                cestice[j][k] = cestice[j][k] + brzine[j][k]

                #provjeravamo da li su unutar dozvoljenog opsega
                #ako nisu vracamo ih na granicu opsega
                if cestice[j][k] > opseg[1]:
                    cestice[j][k] = copy.deepcopy(opseg[1])
                if cestice[j][k] < opseg[0]:
                    cestice[j][k] = copy.deepcopy(opseg[0])

        i = i + 1

    return rjesenje

#Graficki prikaz funkcija
def nacrtajGraf(f, naziv, xZvjezdicax, xZvjezdicay, PSO, start, end, num, levels):
    plt.figure()
    x1 = np.linspace(start, end, num)
    x2 = np.linspace(start, end, num)
    X1, X2 = np.meshgrid(x1, x2)
    Y = f(X1, X2)
    ax = plt.axes(projection='3d')
    ax.contour(X1, X2, Y, levels, cmap='binary')
    ax.scatter(xZvjezdicax, xZvjezdicay, f(xZvjezdicax, xZvjezdicay), color='red', marker='o')
    ax.scatter(PSO[0], PSO[1], f(PSO[0], PSO[1]), color='blue', marker='X', s=100)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title(naziv)
    plt.show()


#Parametri za funkcije
N = 1000
M = 40
eps = 0.00001

W = [0.7, 0.8]
C1 = [r.uniform(0, 1.47), r.uniform(0, 1.62)]
C2 = [r.uniform(0, 1.47), r.uniform(0, 1.62)]
P = [W, C1, C2]


#Paraboloid
opseg_paraboloid = (-5, 5)
x0_paraboloid = [-4, 1]
x_zvjezdica_paraboloid = [3, -1]
paraboloid = lambda x, y: (x - 3)**2 + (y + 1)**2
pso_paraboloid = PSO(paraboloid, opseg_paraboloid, N, M, eps, P)
#print("Paraboloid: ", pso_paraboloid)
nacrtajGraf(paraboloid, 'Paraboloid', x_zvjezdica_paraboloid[0], x_zvjezdica_paraboloid[1],
            pso_paraboloid, -10, 10, 100, 50)

#Rastrigin
opseg_rastrigin = (-5.12, 5.12)
x0_rastrigin = [3, -3]
x_zvjezdica_rastrigin = [0, 0]
rastrigin = lambda x, y: 20 + (x*x - 10*np.cos(2*np.pi*x)) + (y*y - 10*np.cos(2*np.pi*y))
pso_rastrigin = PSO(rastrigin, opseg_rastrigin, N, M, eps, P)
#print("Rastrign: ", pso_rastrigin)
nacrtajGraf(rastrigin, 'Rastrigin', x_zvjezdica_rastrigin[0], x_zvjezdica_rastrigin[1],
            pso_rastrigin, -10, 10, 100, 50)

#Drop-Wave
opseg_dropwave = (-5.12, 5.12)
x0_dropwave = [-4, 5]
x_zvjezdica_dropwave = [0, 0]
drop_wave = lambda x, y: -(1 + np.cos(12*np.sqrt(x**2 + y**2))) / (0.5*(x**2 + y**2) + 2)
pso_dropwave = PSO(drop_wave, opseg_dropwave, N, M, eps, P)
#print("Drop-Wawe: ", pso_dropwave)
nacrtajGraf(drop_wave, 'Drop-Wave', x_zvjezdica_dropwave[0], x_zvjezdica_dropwave[1],
            pso_dropwave, -10, 10, 100, 50)

#Holder Table
opseg_holdertable = (-10, 10)
x0_holdertable = [8, -8]
x_zvjezdica_holdertable = [8.05502, -9.66459]
holderTable = lambda x, y: -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x * x + y * y) / np.pi))))
pso_holdertable = PSO(paraboloid, opseg_holdertable, N, M, eps, P)
#print("Holder Table: ", pso_holdertable)
nacrtajGraf(holderTable, 'Holder Table', x_zvjezdica_holdertable[0], x_zvjezdica_holdertable[1],
            pso_holdertable, -10, 10, 900, 800)

#Schaffer N.2
opseg_schaffer = (-100, 100)
x0_schaffer = [1, 1]
x_zvjezdica_schaffer = [0, 0]
schaffer = lambda x, y: 0.5 + (np.sin(x ** 2 - y ** 2) ** 2 - 0.5) / ((1 + 0.001 * (x ** 2 + y ** 2)) ** 2)
pso_schaffer = PSO(schaffer, opseg_schaffer, N, M, eps, P)
#print("Schaffer N.2: ", pso_schaffer)
nacrtajGraf(schaffer, 'Schaffer N.2', x_zvjezdica_schaffer[0], x_zvjezdica_schaffer[1],
            pso_schaffer, -10, 10, 100, 50)

#Matyas
opseg_matyas = (-10, 10)
x0_matyas = [1, 1]
x_zvjezdica_matyas = [0, 0]
matyas = lambda x, y: 0.26 * (x**2 + y**2) - 0.48 * x * y
pso_matyas = PSO(matyas, opseg_matyas, N, M, eps, P)
#print("Mateyas: ", pso_matyas)
nacrtajGraf(matyas, 'Matyas', x_zvjezdica_matyas[0], x_zvjezdica_matyas[1],
            pso_matyas, -10, 10, 100, 50)



