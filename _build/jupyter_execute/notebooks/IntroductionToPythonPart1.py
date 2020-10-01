**MATH 307   Applications of Linear Algebra**

In this course we will use Python to solve and analyse linear equation with the help of a computer. 
This small *Introduction* will show you how you can use Python on your computer and give you a first glimps of what it is capable off. There are many Python tutorials in the internet and the best way to learn a (new) programming *language* is to get out there and play around with it. 

We will mainly use the Python libraries NumPy, Matplotlib, SciPy and SymPy. In order to use them we need to import them 

## Part 1: SymPy

import sympy as sy

Now we can use all the commands and function in the SymPy (symbolic computations) library in the following way

M = sy.Matrix([[2,1,1],[2,0,2],[4,3,4]])

M

We have defined a symbolic Matrix. Why symbolic? We will come to that later. Lets first see what we can do with this Matrix. 

M.shape

M.row(0)

M.col(-1)

M.col_del(1)

M

M = M.row_insert(2, Matrix([[-1,-2]]))

An error, such an error tells us a lot! Lets see what the problem is. 
It does not know the name 'Matrix' and it is clear why, we did not specify that Matrix belongs to the SymPy library. 
Lets try it again 

M = M.row_insert(2, sy.Matrix([[-1, -2]]))

M

M = M.col_insert(2, sy.Matrix([6, 7 ,8]))

Oh no, another error. We can see that the size of the column we want to insert is wrong. But why? 

M.shape

Ah our matrix is now 4 by 2. So we actually need to insert columns of size 4 

M = M.col_insert(2, sy.Matrix([6, 7 ,8, 0]))

M

**Lets do some Math:**

M = sy.Matrix([[2,1,1],[2,0,2],[4,3,4]])
N = sy.Matrix([[3,0,2],[1,-3,4],[-2,1,0]])

Addition and substraction of Matrices

M+N

M-N

Scalar Multiplication 

3*M

Calculations with Matrices

2*M - 6*N

Matrix Multiplication

M*M

M**2

The Inverse of a matrix

V = N**-1

N*V

M.col_del(0)

M

Transpose of a Matrix

M.T

*More operations for matrices*

sy.eye(4)

sy.zeros(2,3)

sy.ones(3,2)

sy.diag(1,2,3)

M = sy.Matrix([[2,1,1],[2,0,2],[4,3,4]])

M

M.det()

M.rref()

Note that this command returns a Matrix and a tuple of indices of the pivot columns

(M1,T1) = M.rref()

M1



T1

This is a boring example, lets take a more intersting one: 

N = sy.Matrix([[2,-4,-3,-1],[1,-2,4,5],[-1,2,1,0]])

(N1,T1) = N.rref()

N1

T1

N.nullspace()

N.columnspace()

### From last time

Last lecture we gave you some linear systems to solve at home. By now we have already seen how we can use the Sym.py library to do this with a computer. 

Why don't you try this out now for yourself. I will be here to answer questions. At the end of this lecture I will present my solutions. 

**Exercises**

1. Find the general solution of the linear system:  $$
\begin{array}{ r @{{}={}} r  >{{}}c<{{}} r  >{{}}c<{{}}  r }
x_1 &-& 3x_2 &+& 4x_3 &=&  1 \\
-2x_1 &+&  5x_2 &-& 7x_3 &=& 1 \\
x_1 &-& 5x_2 &+& 8x_3 &=& 5 \\
\end{array}
$$


2. Does the matrix equation $Ax = b$ have a solution? If yes, write down the general solution of the system:  
$$ 
A = 
\begin{bmatrix}
2 & -8 & 4 & 2 \\
1 & -3 & 0 & 2 \\
-1 & 2 & 2 & -4 \\
-3 & 11 & -4 & 2
\end{bmatrix}\,, \quad 
b = \begin{bmatrix}
1 \\ 2 \\ 5 \\ 2
\end{bmatrix}
$$

3. What about the equation $Bx = c$? If it has a solution, write it down in the most general form:    
$$
B = \begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & -1 & -1 \\
1 & -1 & 0 & 0 \\
0 & 0 & 1 & 1
\end{bmatrix}\,, \quad 
c = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}
$$

Use SymPy solve these exercises. 

### Can we always use SymPy?

We see that we can already do a lot with SymPy. However, as was said in the beginning this library is for symbolic calculations, not numerical ones. For example 

sy.sqrt(8)

That is the correct answer, but not a numerical value we can work with in a computer since $\sqrt 2$ is irrational

import numpy as np

np.sqrt(8)

This is an approximate solution, however one which is accurate enough on a computer where we will always make errors due to the machine precision. 

**Next time we will learn more about NumPy**