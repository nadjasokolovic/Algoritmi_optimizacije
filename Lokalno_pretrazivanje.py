import math as m
import numpy as np
import random as r
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Jednostavno lokalno pretrazivanje
def LS(f, pocetakOpsega, krajOpsega, x0_x, x0_y, delta_x, N, eps):
    if (daLiJeUOpsegu(pocetakOpsega, krajOpsega, x0_x, x0_y)):
        x1 = x0_x
        x2 = x0_y
        for i in range(0, N):
            okolneTacke = []
            #na osnovu delte formiramo listu okolnih tacaka
            okolneTacke.append((x1, x2 - delta_x))
            okolneTacke.append((x1, x2 + delta_x))
            okolneTacke.append((x1 - delta_x, x2))
            okolneTacke.append((x1 + delta_x, x2))
            okolneTacke.append((x1 - delta_x, x2 - delta_x))
            okolneTacke.append((x1 - delta_x, x2 + delta_x))
            okolneTacke.append((x1 + delta_x, x2 - delta_x))
            okolneTacke.append((x1 + delta_x, x2 + delta_x))
            
            promjena = 0
            for j in range(0, 4): 
                #ako je tacka u opsegu i ako dobijamo bolju vrijednost funkcije
                if (daLiJeUOpsegu(pocetakOpsega, krajOpsega, okolneTacke[j][0], okolneTacke[j][1]) and f(okolneTacke[j][0], okolneTacke[j][1]) < f(x1, x2)):
                    x1 = okolneTacke[j][0]
                    x2 = okolneTacke[j][1]
                    promjena = 1
                
            #ako nije pronadjena niti jedna bolja tacka       
            if promjena == 0:
                break
            
        return x1, x2
    
    return "Početna tačka nije unutar opsega"

def daLiJeUOpsegu (pocetak, kraj, x1, x2):
    return x1 >= pocetak[0] and x2 >= pocetak[1] and x1 <= kraj[0] and x2 <= kraj[1]

def AzuriranjeMemorije(noviLokOptimum, posjeceneTacke):
    posjeceneTacke.append(noviLokOptimum)
    return posjeceneTacke

def Perturbacija (x, pocetakOpsega, krajOpsega):
    x1 = r.uniform(pocetakOpsega[0], krajOpsega[0])
    y1 = r.uniform(pocetakOpsega[1], krajOpsega[1])
     
    while (x1, y1) == x: 
       x1 = r.uniform(pocetakOpsega[0], krajOpsega[0])
       y1 = r.uniform(pocetakOpsega[1], krajOpsega[1])
       
    return x1, y1

#Ponavljano lokalno pretrazivanje
def ILS(f, pocetakOpsega, krajOpsega, x0_x, x0_y, delta_x, N, eps, M): 
    posjeceneTacke = []
    lokalniOptimum = LS(f, pocetakOpsega, krajOpsega, x0_x, x0_y, delta_x, N, eps)
    for i in range (0, M):       
        x_novo = Perturbacija (lokalniOptimum, pocetakOpsega, krajOpsega)
        noviLokOptimum = LS(f, pocetakOpsega, krajOpsega, x_novo[0], x_novo[1], delta_x, N, eps)
        if i == M - 1:
            tacke = AzuriranjeMemorije(noviLokOptimum, posjeceneTacke)
            globalniMin = tacke[0]
            for j in range (1, len(tacke)):
                if f(tacke[j][0], tacke[j][1]) < f(globalniMin[0], globalniMin[1]):
                    globalniMin = (tacke[j][0], tacke[j][1])
        else:
            AzuriranjeMemorije(noviLokOptimum, posjeceneTacke)
            
    return globalniMin

#Testne funkcije
def f1(x1, x2):
    return (x1 - 3)**2 + (x2 + 1)**2

def f2(x1, x2):
    return 20 + (x1**2 - 10*np.cos(2 * np.pi * x1) + x2**2 - 10 * np.cos(2 * np.pi * x2)) 

def f3(x1, x2):
    return -((1 + np.cos(12 * np.sqrt(x1**2 + x2**2))) / (0.5 * (x1**2 + x2**2) + 2))
   
def f4(x1, x2):
    return -(np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - (np.sqrt(x1**2 + x2**2)) / (np.pi)))))
    

print(LS(f1, (-5,-5), (5,5), 0, 0, 0.1, 100, 0.0001))
print(LS(f2, (-5.12, -5.12), (5.12,5.12), 1, 1, 0.1, 100, 0.0001))
print(LS(f2, (-5.12, -5.12), (5.12,5.12), 0, 0, 0.1, 100, 0.0001))
print(LS(f3, (-5.12,-5.12), (5.12,5.12), 2, 3, 0.1, 100, 0.0001))
print(LS(f3, (-5.12,-5.12), (5.12,5.12), 0, 0, 0.1, 100, 0.0001))
print(LS(f4, (-10, -10), (10, 10), 0, 0, 0.1, 100, 0.0001))
print(LS(f4, (-10, -10), (10, 10), 7, 8, 0.1, 100, 0.0001))

#Graficki prikaz funkcija

#Testiranje prve funkcije
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f1(X1, X2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=20.) 
ax.contour(X1, X2, Y, 50, cmap='binary')
tacka = LS(f1, (-5,-5), (5,5), 0, 0, 0.1, 100, 0.0001)
ax.scatter(tacka[0], tacka[1], f1(tacka[0], tacka[1]), color='blue', marker='x', s=150)
ax.scatter(3, -1, f1(3, -1), color='red', marker='o')
iTacka = ILS(f1, (-5, -5), (5, 5), 0, 0, 0.1, 100, 0.0001, 100)
print(iTacka)
ax.scatter(iTacka[0], iTacka[1], f1(iTacka[0], iTacka[1]), color='green', marker='x', s=80)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$');
ax.set_title('$f(x_1,x_2)=(x_1-3)^2+(x_2+1)^2$')
plt.show()

#Testiranje druge funkcije
x1 = np.linspace(-5.12, 5.12, 100)
x2 = np.linspace(-5.12, 5.12, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f2(X1, X2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=10.)
ax.contour(X1, X2, Y, 50, cmap='binary')
tacka = LS(f2, (-5.12, -5.12), (5.12,5.12), 1, 1, 0.1, 100, 0.0001)
ax.scatter(tacka[0], tacka[1], f2(tacka[0], tacka[1]), color='blue', marker='x', s=15)
iTacka = ILS(f2, (-5.12, -5.12), (5.12,5.12), 1, 1, 0.1, 100, 0.0001, 100)
print(iTacka)
ax.scatter(iTacka[0], iTacka[1], f2(iTacka[0], iTacka[1]), color='green', marker='x')
ax.scatter(0, 0, f2(0,0), color='red', marker='o') 
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$');
ax.set_title('$f(x_1,x_2)=20 + (x1^2 - 10cos(2\pi*x1) + x2^2 - 10cos(2\pi*x2))$')
plt.show()

#Testiranje trece funkcije
x1 = np.linspace(-5.12, 5.12, 100)
x2 = np.linspace(-5.12, 5.12, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f3(X1, X2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=5.) 
ax.contour(X1, X2, Y, 50, cmap='binary')
tacka = LS(f3, (-5.12,-5.12), (5.12,5.12), 2, 3, 0.1, 100, 0.0001)
ax.scatter(tacka[0], tacka[1], f3(tacka[0], tacka[1]), color='blue', marker='x', s=50)
ax.scatter(0, 0, f3(0,0), color='red', marker='o')
iTacka = ILS(f3, (-5.12, -5.12), (5.12,5.12), 2, 3, 0.1, 100, 0.0001, 100)
print(iTacka)
ax.scatter(iTacka[0], iTacka[1], f3(iTacka[0], iTacka[1]), color='green', marker='x')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$');
ax.set_title('$f(x_1,x_2)=-((1 + cos(12\sqrt{x1^2 + x2^2}))/(0.5(x1^2 + x2^2) + 2))$')
plt.show()

#Testiranje cetvrte funkcije
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f4(X1, X2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=35.)
ax.contour(X1, X2, Y, 50, cmap='binary')
tacka = LS(f4, (-10, -10), (10, 10), 0, 0, 0.1, 100, 0.0001) 
ax.scatter(tacka[0], tacka[1], f4(tacka[0], tacka[1]), color='blue', marker='x', s=120)
ax.scatter(8.05502, 9.66459, f4(8.05502, 9.66459), color='red', marker='o')
iTacka = ILS(f4, (-10, -10), (10, 10), 0, 0, 0.1, 100, 0.0001, 100) 
print(iTacka)
ax.scatter(iTacka[0], iTacka[1], f4(iTacka[0], iTacka[1]), color='green', marker='x', s=80)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$');
ax.set_title('$f(x_1,x_2)=-|(sin(x1)cos(x2)exp(|1 - (\sqrt{x1**2+x2**2})/(\pi)|))|$')
plt.show()

