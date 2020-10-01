## The LU decomposition in python

The routines to compute the A=PLU factorization are in the scipy library. See
[lu_factor](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html#scipy.linalg.lu_factor)
and [lu_solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html#scipy.linalg.lu_solve).


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

Start with the matrix
$$
A=\begin{bmatrix}
3&2&3\\1&1&1\\1&0&1\\
\end{bmatrix}
$$
and run lu_factor

A=np.array([[3,2,3],[1,1,1],[1,0,1]])
lu, piv = la.lu_factor(A)
print(lu)
print(piv)

**QUESTION 1**: Read the documentation and explain how to recover $A$ from the output of lu_factor(A). Illustrate with the example $A=\begin{bmatrix}
3&2&3\\1&1&1\\1&0&1\\
\end{bmatrix}$



**QUESTION 2**: To solve $Ax=b$ we can use 
 
 
(1)
x=la.solve(A,b) 

Or, we can use the LU factorization

(2)
lu, piv = la.lu_factor(A)
x = la.lu_solve((lu,piv),b)

Compare the time it takes to run (1) and (2) when A and b are large random matrices/vectors. Use np.random.rand(N,N) to obtain matrices whose entries are uniformly distributed on $[0,1]$

