import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def SA(f, pocetakOpsega, krajOpsega, x0_x, x0_y, delta_x, N, eps, T0):
    pocetnaTemperatura = T0
    krajnjaTemperatura = 0
    
    trenutnaTemperatura = pocetnaTemperatura

    x1 = x0_x
    x2 = x0_y
    trenutniMin = f(x1, x2)
    najboljeRjesenje = (x1, x2)
    najboljaF = f(najboljeRjesenje[0], najboljeRjesenje[1])
    brIteracije = 0
    while brIteracije < N and trenutnaTemperatura > krajnjaTemperatura:
        brIteracije += 1
        for k in range (0, 8):
            susjednaTacka = random.choice(generisiOkolinu(najboljeRjesenje[0], najboljeRjesenje[1], delta_x))
            if tackaUOpsegu(pocetakOpsega, krajOpsega, susjednaTacka[0], susjednaTacka[1]) == 0:
                #provjeravamo da li postoji okolna tacka koja je u opsegu
                postoji = 0
                okolneTacke = generisiOkolinu(najboljeRjesenje[0], najboljeRjesenje[1], delta_x)
                for i in range (0,8):
                    if (tackaUOpsegu(pocetakOpsega, krajOpsega, okolneTacke[i][0], okolneTacke[i][1])):
                        postoji = 1
                        susjednaTacka = okolneTacke[i]
                        break
                if postoji == 0: #ne postoji nijedna okolna tacka u opsegu
                    #potrebno je tacke koje su van opsega vratiti na rub opsega
                    for j in range (0, 8):
                        #provjeravamo koja je koordinata izvan opsega
                        if okolneTacke[j][0] < pocetakOpsega[0]:
                            okolneTacke[j] = list(okolneTacke[j])
                            okolneTacke[j][0] = pocetakOpsega[0]
                            okolneTacke[j] = tuple(okolneTacke[j])
                        if okolneTacke[j][0] > krajOpsega[0]:
                            okolneTacke[j] = list(okolneTacke[j])
                            okolneTacke[j][0] = krajOpsega[0]
                            okolneTacke[j] = tuple(okolneTacke[j])
                        if okolneTacke[j][1] < pocetakOpsega[1]:
                            okolneTacke[j] = list(okolneTacke[j])
                            okolneTacke[j][1] = pocetakOpsega[1]
                            okolneTacke[j] = tuple(okolneTacke[j])
                        if okolneTacke[j][1] > krajOpsega[1]:
                            okolneTacke[j] = list(okolneTacke[j])
                            okolneTacke[j][1] = krajOpsega[1] 
                            okolneTacke[j] = tuple(okolneTacke[j])
                    susjednaTacka = random.choice(okolneTacke)

            novaF = f(susjednaTacka[0], susjednaTacka[1])
            razlika = novaF - najboljaF
            #odredjujemo vjerovatnocu prihvatanja rjesenja
            if razlika < 0 or random.uniform(0, 1) < math.exp(-math.fabs(razlika) / (trenutnaTemperatura)):
                najboljeRjesenje = (susjednaTacka[0], susjednaTacka[1])
                najboljaF = novaF
        #smanjujemo temperaturu
        trenutnaTemperatura = trenutnaTemperatura*0.9
        if najboljaF < trenutniMin:
            x1 = najboljeRjesenje[0]
            x2 = najboljeRjesenje[1]
            trenutniMin = najboljaF
            
    return x1, x2

    
def generisiOkolinu(x1, x2, delta_x):
    okolneTacke = []
    okolneTacke.append((x1, x2 - delta_x))
    okolneTacke.append((x1, x2 + delta_x))
    okolneTacke.append((x1 - delta_x, x2))
    okolneTacke.append((x1 + delta_x, x2))
    okolneTacke.append((x1 - delta_x, x2 - delta_x))
    okolneTacke.append((x1 - delta_x, x2 + delta_x))
    okolneTacke.append((x1 + delta_x, x2 - delta_x))
    okolneTacke.append((x1 + delta_x, x2 + delta_x))
    
    return okolneTacke
    
def tackaUOpsegu (pocetak, kraj, x1, x2):
    return x1 >= pocetak[0] and x2 >= pocetak[1] and x1 <= kraj[0] and x2 <= kraj[1]
 
#Definisanje testnih funkcija   
def f1(x1, x2):
    return (x1 - 3)**2 + (x2 + 1)**2

def f2(x1, x2):
    return 20 + (x1**2 - 10*np.cos(2*np.pi*x1) + x2**2 - 10*np.cos(2*np.pi*x2)) 

def f3(x1, x2):
    return -((1 + np.cos(12*np.sqrt(x1**2 + x2**2)))/(0.5*(x1**2 + x2**2) + 2))
   
def f4(x1, x2):
    return -(np.abs(np.sin(x1)*np.cos(x2)*np.exp(np.abs(1 - (np.sqrt(x1**2+x2**2))/(np.pi)))))

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
tacka = SA(f1, (-5,-5), (5,5), -2, 2, 0.5, 5000, 0.0001, 10000)
print(tacka)
ax.scatter(tacka[0], tacka[1], f1(tacka[0], tacka[1]), color='blue', marker='x', s=150)
ax.scatter(3, -1, f1(3, -1), color='red', marker='o')
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
tacka = SA(f2, (-5.12, -5.12), (5.12,5.12), -2.0, 2.0, 0.5, 5000, 0.0001, 10000) #0.5 i 0.5 moze
print(tacka)
ax.scatter(tacka[0], tacka[1], f2(tacka[0], tacka[1]), color='blue', marker='x', s=150)
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
tacka = SA(f3, (-5.12,-5.12), (5.12,5.12), -2, 2, 0.5, 5000, 0.0001, 10000)
print(tacka)
ax.scatter(tacka[0], tacka[1], f3(tacka[0], tacka[1]), color='blue', marker='x', s=150)
ax.scatter(0, 0, f3(0,0), color='red', marker='o')
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
tacka = SA(f4, (-10, -10), (10, 10), -2, 2, 0.5, 5000, 0.0001, 10000) #bilo -9, -7
print(tacka)
ax.scatter(tacka[0], tacka[1], f4(tacka[0], tacka[1]), color='blue', marker='x', s=150)
ax.scatter(8.05502, 9.66459, f4(8.05502, 9.66459), color='red', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$');
ax.set_title('$f(x_1,x_2)=-|(sin(x1)cos(x2)exp(|1 - (\sqrt{x1**2+x2**2})/(\pi)|))|$')
plt.show()


     