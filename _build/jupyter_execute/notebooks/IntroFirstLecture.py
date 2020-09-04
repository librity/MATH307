# MATH 307  --   What is it useful for

## Interpolation of data points

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

Lets imaging you what to interpolate between some given data points for a project with some scientist on the other side of the earth. 

You get the following data point
(x_0, y_0) = (-3, 1) ; (x_1,y_1) = (0, -2) ; (x_2,y_2) = (2, 2)

x = np.array([-1, -0.8, 0])
y = np.array([8, 5.5, 3])
A = np.vander(x,increasing=True)

print(A)

The matrix A gives use a linear system we can solve to find the interpolation between these points 

c = la.solve(A,y)

print(c)

This is the solution of the system Ac = y. Lets see how the plot looks like

T = np.linspace(-1,0,100)
Y = c[0] + c[1]*T + c[2]*T**2 
plt.plot(T,Y,'b-',x,y,'r.',markersize=10)
plt.grid(True)
plt.show()

Then an email reaches you and you get the information that there has to be the additional constraint Y(-1.5) = 0. 
In other words you get an additional data point (x_3, y_3) = (-1/3, 3) 

x = np.array([-1, -0.8, -1/3, 0])
y = np.array([8, 5.5,3,3])
A = np.vander(x,increasing=True)

print(A)

c = la.solve(A,y)

print(c)

T = np.linspace(-1,0,100)
Y = c[0] + c[1]*T + c[2]*T**2 + c[3]*T**3 
plt.plot(T,Y,'b-',x,y,'r.',markersize=10)
plt.grid(True)
plt.show()

That is still all fine. Now lets assume we get the following set of data points which was constructed by a lot of measurements with a very faulty measurement device.

c0 = 3
c1 = 5
c2 = 8
c3 = -2
N = 1000
t = np.random.rand(N) - 1 # Random numbers in the interval (-1,1)
noise = np.random.randn(N)
y = c0 + c1*t + c2*t**2 + c3*t**3 + noise
plt.scatter(t,y,alpha=0.5,lw=0,s=10);
plt.show()

The plot of the datapoints has resemblence with the previous plot. But how do we find a solution to describe these data points with minimal error.

A = np.column_stack([np.ones(N),t,t**2])

We solve $\left( A^T A \right) \mathbf{c} = \left( A^T \right) \mathbf{y}$ which gives us the least-square solution

c = la.solve(A.T @ A,A.T @ y)


ts = np.linspace(-1,0,20)
ys = c[0] + c[1]*ts + c[2]*ts**2 
plt.plot(ts,ys,'r',linewidth=4)
plt.scatter(t,y,alpha=0.5,lw=0)
plt.show()

### Image deblurring

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.io import loadmat

kitten = plt.imread('data/kitten.jpg').astype(np.float64)

N = 256
c = np.zeros(N)
s = 5
c[:s] = (s - np.arange(0,s))/(3*s)
Ac = la.toeplitz(c)
r = np.zeros(N)
s = 20
r[:s] = (s - np.arange(0,s))/(3*s)
Ar = la.toeplitz(r)

B = Ac@kitten@Ar.T + 0.01*np.random.randn(256,256)

plt.imshow(B,cmap='gray')
plt.show()

The image of the kitten is blurred by some noise $E$, so that 

$$
A_c X A_r^T = B + E
$$

How do we find $X$, i.e. the unblurred image of the kitten?

We compute using the truncated pseudoinverse

$$
X = (A_c)_k^+ B (A_r^T)_k^+
$$

Pc,Sc,QTc = la.svd(Ac)
Pr,Sr,QTr = la.svd(Ar)

k = 50
Dc_k_plus = np.hstack([1/Sc[:k],np.zeros(N-k)])
Dr_k_plus = np.hstack([1/Sr[:k],np.zeros(N-k)])
Ac_k_plus = QTc.T @ np.diag(Dc_k_plus) @ Pc.T
Ar_k_plus = Pr @ np.diag(Dr_k_plus) @ QTr
X = Ac_k_plus @ B @ Ar_k_plus

plt.imshow(X,cmap='gray')
plt.show()

k = 100
Dc_k_plus = np.hstack([1/Sc[:k],np.zeros(N-k)])
Dr_k_plus = np.hstack([1/Sr[:k],np.zeros(N-k)])
Ac_k_plus = QTc.T @ np.diag(Dc_k_plus) @ Pc.T
Ar_k_plus = Pr @ np.diag(Dr_k_plus) @ QTr
X = Ac_k_plus @ B @ Ar_k_plus

plt.imshow(X,cmap='gray')
plt.show()

k = 200
Dc_k_plus = np.hstack([1/Sc[:k],np.zeros(N-k)])
Dr_k_plus = np.hstack([1/Sr[:k],np.zeros(N-k)])
Ac_k_plus = QTc.T @ np.diag(Dc_k_plus) @ Pc.T
Ar_k_plus = Pr @ np.diag(Dr_k_plus) @ QTr
X = Ac_k_plus @ B @ Ar_k_plus

plt.imshow(X,cmap='gray')
plt.show()

plt.imshow(kitten,cmap='gray')
plt.show()

