#Newton-Rapsonov algoritam
from sympy import *
import math as m
import numpy as np
import matplotlib.pyplot as plt
x = Symbol('x')

def NR (f, df, ddf, x0, N, eps):    
    if NR.n < N: #ako nije dostignut maksimalan broj operacija
        NR.n += 1
        if np.abs(ddf(x0)) < eps:
            x0 = x0 + 0.5 #kako bismo izbjegli dijeljenje sa nulom
        xn = x0 - (df(x0) / ddf(x0))
        
        if np.abs(f(xn) - f(x0)) < eps: #uslov zaustavljanja
            return xn        
        
        return NR(f, df, ddf, xn, N, eps)
     
    return x0  

#Testiranje za f(x) = 3x^2 − 1, x0 pripada {1, 10}
NR.n = 0
f1 = 3*x*x - 1
df1 = f1.diff(x)
ddf1 = df1.diff(x)
f1 = lambdify(x, f1)
df1 = lambdify(x, df1)
ddf1 = lambdify(x, ddf1)

tmp = NR(f1, df1, ddf1, 1, 100, 0.000001)
print('(x,y): ({},{})'.format(tmp, f1(tmp)))
tmp = NR(f1, df1, ddf1, 10, 100, 0.000001)
print('(x,y): ({},{})'.format(tmp, f1(tmp)))

#Testiranje za f(x) = −(16x^2 − 24x + 5)e^−x, x0 pripada {0.5, 2}
NR.n = 0
f2 = -(16 * x * x - 24 * x + 5) * np.e**(-x)
df2 = f2.diff(x)
ddf2 = df2.diff(x)
f2 = lambdify(x, f2)
df2 = lambdify(x, df2)
ddf2 = lambdify(x, ddf2)

tmp = NR(f2, df2, ddf2, 0.5, 100, 0.000001)
print('(x,y): ({},{})'.format(tmp, f2(tmp)))
tmp = NR(f2, df2, ddf2, 2, 100, 0.000001)
print('(x,y): ({},{})'.format(tmp, f2(tmp)))

#Testiranje za f(x) = sin(x) + sin((10/3)/x), x0 pripada {3, 6, 7}
NR.n = 0
f3 = lambda a: np.sin(a) + np.sin((10/3) * a)
df3 = lambda a: 10*np.cos((10/3)*a)/3 + np.cos(a)
ddf3 = lambda a: -((100/9)*np.sin((10/3)*a)) - np.sin(a)

tmp = NR(f3, df3, ddf3, 3, 100, 0.000001)
print('(x,y): ({},{})'.format(tmp, f3(tmp)))
tmp = NR(f3, df3, ddf3, 6, 100, 0.000001)
print('(x,y): ({},{})'.format(tmp, f3(tmp)))
tmp = NR(f3, df3, ddf3, 7, 100, 0.000001)
print('(x,y): ({},{})'.format(tmp, f3(tmp)))


#Testiranje za f(x) = e^−x, x0 pripada {1, 10}
NR.n = 0
f4 = np.e**(-x)
df4 = f4.diff(x)
ddf4 = df4.diff(x)
f4 = lambdify(x, f4)
df4 = lambdify(x, df4)
ddf4 = lambdify(x, ddf4)
tmp  = NR(f4, df4, ddf4, 1, 100, 0.00001)
print('(x,y) = (', tmp, ',', f4(tmp), ')')
tmp  = NR(f4, df4, ddf4, 10, 100, 0.00001)
print('(x,y) = (', tmp, ',', f4(tmp), ')')

#Graficki prikazi funkcija

#Testiranje prve funkcije 
x = np.arange(-3, 3, 0.01) 
y = 3*x*x - 1 #definisemo funkciju
plt.figure()
plt.plot(x, y, color='black') 
plt.ylim(-3, 3)
x0 = NR(f1, df1, ddf1, 1, 100, 0.00001) 
plt.plot(x0, f1(x0), color='red', marker='o') 
plt.plot(x0, f1(x0), color='green', marker='x')
x0 = NR(f1, df1, ddf1, 10, 100, 0.00001)
plt.plot(x0, f1(x0), color='green', marker='x')
plt.grid(True)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$y=3x*x - 1$')
plt.show()

#Testiranje druge funkcije
x = np.arange(-4, 25, 0.01)
y = -(16*x*x -24*x + 5) * np.e**(-x)
plt.figure()
plt.plot(x, y, color='black')
plt.ylim(-10, 4)
x0 = NR(f2, df2, ddf2, 0.5, 100, 0.00001)
x0_1 = NR(f2, df2, ddf2, 2, 100, 0.00001)
x0_2 = NR(f2, df2, ddf2, 10, 100, 0.00001)
plt.plot(x0, f2(x0), color='green', marker='x', markerSize=15)
plt.plot(x0_1, f2(x0_1), color='green', marker='x', markerSize=15)
plt.plot(x0, f2(x0), color='red', marker='o') 
plt.plot(x0_1, f2(x0_1), color='red', marker='o') 
plt.plot(x0_2, f2(x0_2), color='red', marker='o') 
plt.grid(True)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$y=-(16*x*x -24*x + 5) * e**(-x)$')
plt.show()

#Testiranje trece funkcije
x = np.arange(-2, 8, 0.01)
y = np.sin(x) + np.sin((10/3) * x)
plt.figure()
plt.plot(x, y, color='black')
plt.ylim(-2, 2)
x0 = NR(f3, df3, ddf3, 0, 100, 0.000001)
x0_1 = NR(f3, df3, ddf3, 6, 100, 0.000001)
x0_2 = NR(f3, df3, ddf3, 7, 100, 0.000001)
plt.plot(x0, f3(x0), color='green', marker='x', markerSize=15) 
plt.plot(x0_1, f3(x0_1), color='green', marker='x', markerSize=15)
plt.plot(x0_2, f3(x0_2), color='green', marker='x', markerSize=15)
plt.plot(x0, f3(x0), color='red', marker='o') 
plt.plot(x0_1, f3(x0_1), color='red', marker='o')
plt.plot(x0_2, f3(x0_2), color='red', marker='o')
plt.grid(True)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$\sin(x) + sin(10*x/3)$')
plt.show()

#Testiranje cetvrte funkcije
x = np.arange(-4, 15, 0.01)
y = np.e**(-x)
plt.figure()
plt.plot(x, y, color='black')
plt.ylim(-1, 5)
x0 = NR(f4, df4, ddf4, 1, 3, 0.00001)
x0_1 = NR(f4, df4, ddf4, 10, 100, 0.00001)
plt.plot(x0, f4(x0), color='red', marker='o') 
plt.plot(x0, f4(x0), color='green', marker='x', markerSize=15)
plt.plot(x0_1, f4(x0_1), color='green', marker='x', markerSize=40)
plt.grid(True)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('$y=e**(-x)$')
plt.show()




