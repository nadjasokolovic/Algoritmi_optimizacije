import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def TS(f, pocetakOpsega, krajOpsega, x0_x, x0_y, delta_x, N, eps, L):
    tabuLista = []
    if (tackaUOpsegu(pocetakOpsega, krajOpsega, x0_x, x0_y)):
        x1 = x0_x
        x2 = x0_y
        najboljeRjesenje = (x1, x2)
        #u svrhu boljih performansi programa pocetnu tacku dodajemo u tabu listu
        tabuLista.append((x1, x2))
        for i in range(0, N):
            okolneTacke = []
            #formiramo okolinu tacke, tj. tacke u kojima cemo vrsiti pretrazivanje
            okolneTacke.append((x1, x2 - delta_x))
            okolneTacke.append((x1, x2 + delta_x))
            okolneTacke.append((x1 - delta_x, x2))
            okolneTacke.append((x1 + delta_x, x2))
            okolneTacke.append((x1 - delta_x, x2 - delta_x))
            okolneTacke.append((x1 - delta_x, x2 + delta_x))
            okolneTacke.append((x1 + delta_x, x2 - delta_x))
            okolneTacke.append((x1 + delta_x, x2 + delta_x))
            postojiDozvoljenaTacka = 0
            for j in range (0, 8):
                #provjeravamo da li postoji neka dozvoljena tacka
                #tacka je dozvoljena ako se nalazi unutar opsega i ako se ne nalazi u tabu listi
                if (tackaUOpsegu(pocetakOpsega, krajOpsega, okolneTacke[j][0], okolneTacke[j][1]) and tabuLista.count(okolneTacke[j]) == 0):
                    x1 = okolneTacke[j][0] 
                    x2 = okolneTacke[j][1]
                    postojiDozvoljenaTacka = 1
                    break; #kada nadjemo prvu dozvoljenu tacku prekidamo pretragu
                    
            if postojiDozvoljenaTacka == 0: #ako nema dozvoljene tacke
                print("Nema dozvoljenih tacaka")
                uOpsegu = 0;
                for j in range(0, 8):
                    #prva ideja jeste da pronadjemo prvu tacku iz okoline koja je unutar dozvoljenog opsega
                    if (tackaUOpsegu(pocetakOpsega, krajOpsega, okolneTacke[j][0], okolneTacke[j][1])):
                        x1 = okolneTacke[j][0]
                        x2 = okolneTacke[j][1]
                        uOpsegu = 1
                        break;
                if uOpsegu == 0: #ako ne postoji tacka iz okoline koja je u opsegu
                    print ("ELSE")
                    for j in range (0, 8):
                        #druga ideja je provjeriti koja koordinata je van opsega te je vratiti na rub opsega
                        if okolneTacke[j][0] < pocetakOpsega[0]:
                            okolneTacke[j][0] = pocetakOpsega[0]
                        if okolneTacke[j][0] > krajOpsega[0]:
                            okolneTacke[j][0] = krajOpsega[0]
                        if okolneTacke[j][1] < pocetakOpsega[1]:
                            okolneTacke[j][1] = pocetakOpsega[1]
                        if okolneTacke[j][1] > krajOpsega[1]:
                            okolneTacke[j][1] = krajOpsega[1]
                        #provjeravamo da li je tacka koju smo pomakli u opsegu
                        if (tackaUOpsegu(pocetakOpsega, krajOpsega, okolneTacke[j][0], okolneTacke[j][1])):
                            x1 = okolneTacke[j][0]
                            x2 = okolneTacke[j][1]
                            break; #sad imamo neku tacku koja je u opsegu
                            
                if (f(x1, x2) < f(najboljeRjesenje[0], najboljeRjesenje[1])):
                    najboljeRjesenje = (x1, x2)
                #ova tacka je vec u tabuListi, pa ne trebamo vrsiti azuriranje liste
               
            else:
                for j in range(0, 8):  
                    if (tackaUOpsegu(pocetakOpsega, krajOpsega, okolneTacke[j][0], okolneTacke[j][1]) and tabuLista.count(okolneTacke[j]) == 0 and f(okolneTacke[j][0], okolneTacke[j][1]) < f(x1, x2)):
                        x1 = okolneTacke[j][0]
                        x2 = okolneTacke[j][1]
                if (f(x1, x2) < f(najboljeRjesenje[0], najboljeRjesenje[1])):
                    najboljeRjesenje = (x1, x2)
                #azuriranje tabu liste   
                tabuLista.append((x1, x2))
                azurirajTabuListu(tabuLista, L)
           
        return najboljeRjesenje
    return "Početna tačka nije unutar opsega"
        
        
def tackaUOpsegu (pocetak, kraj, x1, x2):
    return x1 >= pocetak[0] and x2 >= pocetak[1] and x1 <= kraj[0] and x2 <= kraj[1]

def azurirajTabuListu(tabuLista, L):
    #ako je tabu lista popunjena izbacamo prvi element
    if len(tabuLista) > L: 
        tabuLista.pop(0)

#Definicija testnih funkcija 
def f1(x1, x2):
    return (x1 - 3)**2 + (x2 + 1)**2

def f2(x1, x2):
    return 20 + (x1**2 - 10 * np.cos(2 * np.pi * x1) + x2**2 - 10 * np.cos(2 * np.pi * x2)) 

def f3(x1, x2):
    return -((1 + np.cos(12 * np.sqrt(x1**2 + x2**2))) / (0.5 * (x1**2 + x2**2) + 2))
   
def f4(x1, x2):
    return -(np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - (np.sqrt(x1**2 + x2**2)) / (np.pi)))))

#Testiranje prve funkcije
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f1(X1, X2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=20.) 
ax.contour(X1, X2, Y, 50, cmap='binary')
tacka = TS(f1, (-5,-5), (5,5), -2, 2, 0.5, 100, 0.0001, 100)
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
tacka = TS(f2, (-5.12, -5.12), (5.12,5.12), -2, 2, 0.5, 100, 0.0001, 100) 
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
tacka = TS(f3, (-5.12,-5.12), (5.12,5.12), -2, 2, 0.5, 100, 0.0001, 100)
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
tacka = TS(f4, (-10, -10), (10, 10), -2, 2, 0.5, 1000, 0.0001, 100) 
print(tacka)
ax.scatter(tacka[0], tacka[1], f4(tacka[0], tacka[1]), color='blue', marker='x', s=150)
ax.scatter(8.05502, 9.66459, f4(8.05502, 9.66459), color='red', marker='o')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$');
ax.set_title('$f(x_1,x_2)=-|(sin(x1)cos(x2)exp(|1 - (\sqrt{x1**2+x2**2})/(\pi)|))|$')
plt.show()
     

