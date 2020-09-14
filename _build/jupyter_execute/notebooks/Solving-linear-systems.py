# Solving Linear Systems

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
%matplotlib inline

## Linear Systems

A [linear system of equations](https://en.wikipedia.org/wiki/System_of_linear_equations) is a collection of linear equations

\begin{align}
a_{0,0}x_0 + a_{0,1}x_2 + \cdots + a_{0,n}x_n & = b_0 \\\
a_{1,0}x_0 + a_{1,1}x_2 + \cdots + a_{1,n}x_n & = b_1 \\\
& \vdots \\\
a_{m,0}x_0 + a_{m,1}x_2 + \cdots + a_{m,n}x_n & = b_m \\\
\end{align}

In matrix notation, a linear system is $A \mathbf{x}= \mathbf{b}$ where

$$
A = \begin{bmatrix}
a_{0,0} & a_{0,1} & \cdots & a_{0,n} \\\
a_{1,0} & a_{1,1} & \cdots & a_{1,n} \\\
\vdots & & & \vdots \\\
a_{m,0} & a_{m,1} & \cdots & a_{m,n} \\\
\end{bmatrix}
 \ \ , \ \
\mathbf{x} = \begin{bmatrix}
x_0 \\\ x_1 \\\ \vdots \\\ x_n
\end{bmatrix}
 \ \ , \ \
\mathbf{b} = \begin{bmatrix}
b_0 \\\ b_1 \\\ \vdots \\\ b_m
\end{bmatrix} 
$$

## Gaussian elimination

The general procedure to solve a linear system of equation is called [Gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination). The idea is to perform elementary row operations to reduce the system to its row echelon form and then solve.

### Elementary Row Operations

[Elementary row operations](https://en.wikipedia.org/wiki/Elementary_matrix#Elementary_row_operations) include:

1. Add $k$ times row $j$ to row $i$.
2. Multiply row $i$ by scalar $k$.
3. Switch rows $i$ and $j$.

Each of the elementary row operations is the result of matrix multiplication by an elementary matrix (on the left).
To add $k$ times row $i$ to row $j$ in a matrix $A$, we multiply $A$ by the matrix $E$ where $E$ is equal to the identity matrix except the $i,j$ entry is $E_{i,j} = k$. For example, if $A$ is 3 by 3 and we want to add 3 times row 2 to row 0 (using 0 indexing) then

$$
E_1 = \begin{bmatrix}
1 & 0 & 3 \\\
0 & 1 & 0 \\\
0 & 0 & 1
\end{bmatrix}
$$

Let's verify the calculation:

A = np.array([[1,1,2],[-1,3,1],[0,5,2]])
print(A)

E1 = np.array([[1,0,3],[0,1,0],[0,0,1]])
print(E1)

E1 @ A

To multiply $k$ times row $i$ in a matrix $A$, we multiply $A$ by the matrix $E$ where $E$ is equal to the identity matrix except the $,i,j$ entry is $E_{i,i} = k$. For example, if $A$ is 3 by 3 and we want to multiply row 1 by -2 then

$$
E_2 = \begin{bmatrix}
1 & 0 & 0 \\\
0 & -2 & 0 \\\
0 & 0 & 1
\end{bmatrix}
$$

Let's verify the calculation:

E2 = np.array([[1,0,0],[0,-2,0],[0,0,1]])
print(E2)

E2 @ A

Finally, to switch row $i$ and row $j$ in a matrix $A$, we multiply $A$ by the matrix $E$ where $E$ is equal to the identity matrix except $E_{i,i} = 0$, $E_{j,j} = 0$, $E_{i,j} = 1$ and $E_{j,i} = 1$. For example, if $A$ is 3 by 3 and we want to switch row 1 and row 2 then

$$
E^3 = \begin{bmatrix}
1 & 0 & 0 \\\
0 & 0 & 1 \\\
0 & 1 & 0
\end{bmatrix}
$$

Let's verify the calculation:

E3 = np.array([[1,0,0],[0,0,1],[0,1,0]])
print(E3)

E3 @ A

### Implementation

Let's write function to implement the elementary row operations. First of all, let's write a function called `add_rows` which takes input parameters $A$, $k$, $i$ and $j$ and returns the NumPy array resulting from adding $k$ times row $j$ to row $i$ in the matrix $A$. If $i=j$, then let's say that the function scales row $i$ by $k+1$ since this would be the result of $k$ times row $i$ added to row $i$.

def add_row(A,k,i,j):
    "Add k times row j to row i in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    if i == j:
        E[i,i] = k + 1
    else:
        E[i,j] = k
    return E @ A

Let's test our function:

M = np.array([[1,1],[3,2]])
print(M)

add_row(M,2,0,1)

add_row(M,3,1,1)

Let's write a function called `scale_row` which takes 3 input parameters $A$, $k$, and $i$ and returns the matrix that results from $k$ times row $i$ in the matrix $A$.

def scale_row(A,k,i):
    "Multiply row i by k in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    E[i,i] = k
    return E @ A

M = np.array([[3,1],[-2,7]])
print(M)

scale_row(M,3,1)

A = np.array([[1,1,1],[1,-1,0]])
print(A)

scale_row(A,5,1)

Let's write a function called `switch_rows` which takes 3 input parameters $A$, $i$ and $j$ and returns the matrix that results from switching rows $i$ and $j$ in the matrix $A$.

def switch_rows(A,i,j):
    "Switch rows i and j in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    E[i,i] = 0
    E[j,j] = 0
    E[i,j] = 1
    E[j,i] = 1
    return E @ A

A = np.array([[1,1,1],[1,-1,0]])
print(A)

switch_rows(A,0,1)

## Examples

### Find the Inverse

Let's apply our functions to the augmented matrix $[M \ | \ I]$ to find the inverse of the matrix $M$:

M = np.array([[5,4,2],[-1,2,1],[1,1,1]])
print(M)

A = np.hstack([M,np.eye(3)])
print(A)

A1 = switch_rows(A,0,2)
print(A1)

A2 = add_row(A1,1,1,0)
print(A2)

A3 = add_row(A2,-5,2,0)
print(A3)

A4 = switch_rows(A3,1,2)
print(A4)

A5 = scale_row(A4,-1,1)
print(A5)

A6 = add_row(A5,-3,2,1)
print(A6)

A7 = scale_row(A6,-1/7,2)
print(A7)

A8 = add_row(A7,-3,1,2)
print(A8)

A9 = add_row(A8,-1,0,2)
print(A9)

A10 = add_row(A9,-1,0,1)
print(A10)

Let's verify that we found the inverse $M^{-1}$ correctly:

Minv = A10[:,3:]
print(Minv)

result = Minv @ M
print(result)

Success! We can see the result more clearly if we round to 15 decimal places:

np.round(result,15)

### Solve a System

Let's use our functions to perform Gaussian elimination and solve a linear system of equations $A \mathbf{x} = \mathbf{b}$.

Recall the linear system from a previous exercise: 

$$
\begin{array}{ r @{{}={}} r  >{{}}c<{{}} r  >{{}}c<{{}}  r }
x_1 &-& 3x_2 &+& 4x_3 &=&  1 \\
-2x_1 &+&  5x_2 &-& 7x_3 &=& 1 \\
x_1 &-& 5x_2 &+& 8x_3 &=& 5 \\
\end{array}
$$

Back then we used SymPy to solve it

import sympy as sy

A = sy.Matrix([[1,-3,4],[-2,5,-7],[1,-5,8]])
A

b = sy.Matrix([[1],[1],[5]])
b

Oldx = A.inv()*b
Oldx

Let's see what we get now:

A = np.array([[1,-3,4],[-2,5,-7],[1,-5,8]])
print(A)

b = np.array([[1],[1],[5]])

**Note:** We can also directly transform a sy.Matrix into a np.array

A = np.array(A).astype(np.float64)

b = np.array(b).astype(np.float64)

Form the augemented matrix $M$:

M = np.hstack([A,b])
print(M)

Perform row operations:

M1 = add_row(M,2,1,0)
print(M1)

M2 = add_row(M1,-1,2,0)
print(M2)

M3 = scale_row(M2,-1,1)
print(M3)

M4 = add_row(M3,2,2,1)
print(M4)

M5 = scale_row(M4,1/2,2)
print(M5)

M6 = add_row(M5,1,1,2)
print(M6)

M7 = add_row(M6,-M6[0,2],0,2)
print(M7)

M8 = add_row(M7,-M7[0,1],0,1)
print(M8)

Success! The solution of $Ax=b$ is

x = M8[:,3].reshape(3,1)
print(x)

Oldx

Or, we can do it the easy way...

x = la.solve(A,b)
print(x)

## `scipy.linalg.solve`

We are mostly interested in linear systems $A \mathbf{x} = \mathbf{b}$ where there is a unique solution $\mathbf{x}$. This is the case when $A$ is a square matrix ($m=n$) and $\mathrm{det}(A) \not= 0$. To solve such a system, we can use the function [`scipy.linalg.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html).

The function returns a solution of the system of equations $A \mathbf{x} = \mathbf{b}$. For example:

A = np.array([[1,1],[1,-1]])
print(A)

b1 = np.array([2,0])
print(b1)

And solve:

x1 = la.solve(A,b1)
print(x1)

Note that the output $\mathbf{x}$ is returned as a 1D NumPy array when the vector $\mathbf{b}$ (the right hand side) is entered as a 1D NumPy array. If we input $\mathbf{b}$ as a 2D NumPy array, then the output is a 2D NumPy array. For example:

A = np.array([[1,1],[1,-1]])
b2 = np.array([2,0]).reshape(2,1)
x2 = la.solve(A,b2)
print(x2)

Finally, if the right hand side $\mathbf{b}$ is a matrix, then the output is a matrix of the same size. It is the solution of $A \mathbf{x} = \mathbf{b}$ when $\mathbf{b}$ is a matrix. For example:

A = np.array([[1,1],[1,-1]])
b3 = np.array([[2,2],[0,1]])
x3 = la.solve(A,b3)
print(x3)

### Simple Example

Let's compute the solution of the system of equations

\begin{align}
2x + y &= 1 \\\
x + y &= 1
\end{align}

Create the matrix of coefficients:

A = np.array([[2,1],[1,1]])
print(A)

And the vector $\mathbf{b}$:

b = np.array([1,-1]).reshape(2,1)
print(b)

And solve:

x = la.solve(A,b)
print(x)

We can verify the solution by computing the inverse of $A$:

Ainv = la.inv(A)
print(Ainv)

And multiply $A^{-1} \mathbf{b}$ to solve for $\mathbf{x}$:

x = Ainv @ b
print(x)

We get the same result. Success!

### Inverse or Solve

It's a bad idea to use the inverse $A^{-1}$ to solve $A \mathbf{x} = \mathbf{b}$ if $A$ is large. It's too computationally expensive. Let's create a large random matrix $A$ and vector $\mathbf{b}$ and compute the solution $\mathbf{x}$ in 2 ways:

N = 1000
A = np.random.rand(N,N)
b = np.random.rand(N,1)

Check the first entries $A$:

A[:3,:3]

And for $\mathbf{b}$:

b[:4,:]

Now we compare the speed of `scipy.linalg.solve` with `scipy.linalg.inv`:

%%timeit
x = la.solve(A,b)

%%timeit
x = la.inv(A) @ b

Solving with `scipy.linalg.solve` is about twice as fast!

## Exercises

Try the functions 'add_row', 'scale_row' and 'switch_row' for yourself and find the solution for the following system $Ax = b$ with 

$$
A = \begin{bmatrix}
6 & 15 & 1 \\
8 & 7 & 12 \\
2 & 7 & 8 
\end{bmatrix} \,, \quad 
b = \begin{bmatrix}
2 \\ 14 \\ 10 \\
\end{bmatrix}
$$