# Linear Systems of Equations

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

Solve linear systems of equations $A \mathbf{x} = \mathbf{b}$:

* Create NumPy arrays to represent $A$ and $\mathbf{b}$
* Compute the solution $\boldsymbol{x}$ using the SciPy function `scipy.linalg.solve`

[Learn about NumPy arrays](https://www.math.ubc.ca/~pwalls/math-python/scipy/numpy/) and the [SciPy Linear Algebra package](https://www.math.ubc.ca/~pwalls/math-python/linear-algebra/linear-algebra-scipy/).

## Example: Solve $A \mathbf{x} = \mathbf{b}$ with `scipy.linalg.solve`

Compute the solution of the system $A \mathbf{x} = \mathbf{b}$ where

$$
A = \begin{bmatrix} 2 & 1 & 1 \\ 2 & 0 & 2 \\ 4 & 3 & 4 \end{bmatrix}
\hspace{10mm}
\mathbf{b} = \left[ \begin{array}{r} -1 \\ 1 \\ 1 \end{array} \right]
$$

A = np.array([[2,1,1],[2,0,2],[4,3,4]])
b = np.array([[-1],[1],[1]])

print(A)

print(b)

type(b)

x = la.solve(A,b)

print(x)

Due to rounding errors in the computation, our solution $\hat{\mathbf{x}}$ is an approximation of the exact solution

$$
\mathbf{x} = \left[ \begin{array}{r} -7/6 \\-1/3 \\ 5/3 \end{array} \right]
$$

Compute the norm of the residual $\| \mathbf{b} - A \mathbf{x} \|$

r = la.norm(b - A @ x)

print(r)

## Example: Resistor Network

Compute the solution of the system $A \mathbf{x} = \mathbf{b}$ for

$$
A = 
\left[
\begin{array}{cccccccc}
2R & -R & 0 & 0 & \cdots & 0 & 0 & 0 \\
-R & 2R & -R & 0 & & 0 & 0  & 0 \\
0 & -R & 2R & -R & \cdots & 0 & 0 & 0 \\
\vdots &  & \vdots &  & \ddots & & \vdots & \\
0 & 0 & 0 & 0 & \cdots & -R & 2R & -R \\
0 & 0 & 0 & 0 & \cdots & 0 & -R & 2R \\
\end{array}
\right]
\hspace{10mm}
\mathbf{b} = \left[ \begin{array}{r} V \\ \vdots \\ V \end{array} \right]
$$

where $A$ is a square matrix of size $N$, and $R$ and $V$ are some positive constants. The system is a mathematical model of the parallel circuilt

![resistor network](data/circuit.png)

such that the solution consists of the loops currents $i_1,\dots,i_N$.

N = 10
R = 1
V = 1
A1 = 2*R*np.eye(N)
A2 = np.diag(-R*np.ones(N-1),1)
A = A1 + A2 + A2.T
b = V*np.ones([N,1])

print(A)

print(b)

x = la.solve(A,b)

plt.plot(x,'b.')
plt.xlabel('Loop current at index i')
plt.ylabel('Loop currents (Amp)')
plt.show()

print(x)