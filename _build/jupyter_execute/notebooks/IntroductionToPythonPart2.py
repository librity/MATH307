Last time we discussed how we can use the SymPy Library. We saw that it is useful for symbolic calculation but has problems if we are interested in the acutal numeric value.

*Today we talk about NumPy and the linear algebra subpackage SciPy [`scipy.linalg`]*

## Part 2: NumPy

import numpy as np

### NumPy Arrays

Let's begin with a quick review of [NumPy arrays](../../scipy/numpy/). We can think of a 1D NumPy array as a list of numbers. We can think of a 2D NumPy array as a matrix. And we can think of a 3D array as a cube of numbers.
When we select a row or column from a 2D NumPy array, the result is a 1D NumPy array (called a [slice](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing)). This is different from MATLAB where when you select a column from a matrix it's returned as a column vector which is a 2D MATLAB matrix.

It can get a bit confusing and so we need to keep track of the shape, size and dimension of our NumPy arrays.

#### Array Attributes

Create a 1D (one-dimensional) NumPy array and verify its dimensions, shape and size.

a = np.array([1,3,-2,1])
print(a)

Verify the number of dimensions:

a.ndim

Verify the shape of the array:

a.shape

The shape of an array is returned as a [Python tuple](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences). The output in the cell above is a tuple of length 1. And we verify the size of the array (ie. the total number of entries in the array):

a.size

Create a 2D (two-dimensional) NumPy array (ie. matrix):

M = np.array([[1,2],[3,7],[-1,5]])
print(M)

Verify the number of dimensions:

M.ndim

Verify the shape of the array:

M.shape

Finally, verify the total number of entries in the array:

M.size

Select a row or column from a 2D NumPy array and we get a 1D array:

col = M[:,1] 
print(col)

Verify the number of dimensions of the slice:

col.ndim

Verify the shape and size of the slice:

col.shape

col.size

When we select a row of column from a 2D NumPy array, the result is a 1D NumPy array. However, we may want to select a column as a 2D column vector. This requires us to use the [reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) method.

For example, create a 2D column vector from the 1D slice selected from the matrix `M` above:

print(col)

column = np.array([2,7,5]).reshape(3,1)
print(column)

Verify the dimensions, shape and size of the array:

print('Dimensions:', column.ndim)
print('Shape:', column.shape)
print('Size:', column.size)

The variables `col` and `column` are different types of objects even though they have the "same" data.

print(col)

print('Dimensions:',col.ndim)
print('Shape:',col.shape)
print('Size:',col.size)

### Matrix Operations and Functions

#### Arithmetic Operations

Recall that arithmetic [array operations](../scipy/numpy/#operations-and-functions) `+`, `-`, `/`, `*` and `**` are performed elementwise on NumPy arrays. Let's create a NumPy array and do some computations:

M = np.array([[3,4],[-1,5]])
print(M)

M * M

#### Matrix Multiplication

We use the `@` operator to do matrix multiplication with NumPy arrays:

M @ M

Let's compute $2I + 3A - AB$ for

$$
A = \begin{bmatrix}
1 & 3 \\\
-1 & 7
\end{bmatrix}
\ \ \ \
B = \begin{bmatrix}
5 & 2 \\\
1 & 2
\end{bmatrix}
$$

and $I$ is the identity matrix of size 2:

A = np.array([[1,3],[-1,7]])
print(A)

B = np.array([[5,2],[1,2]])
print(B)

I = np.eye(2)
print(I)

2*I + 3*A - A@B

#### Matrix Powers

There's no symbol for matrix powers and so we must import the function `matrix_power` from the subpackage `numpy.linalg`.

from numpy.linalg import matrix_power as mpow

M = np.array([[3,4],[-1,5]])
print(M)

mpow(M,2)

mpow(M,5)

Compare with the matrix multiplcation operator:

M @ M @ M @ M @ M

mpow(M,3)

M @ M @ M

####  Tranpose

We can take the transpose with `.T` attribute:

print(M)

print(M.T)

Notice that $M M^T$ is a symmetric matrix:

M @ M.T

#### Trace

We can find the trace of a matrix using the function `numpy.trace`:

np.trace(A)

## Linear Algebra with SciPy

The main Python package for linear algebra is the SciPy subpackage [`scipy.linalg`](https://docs.scipy.org/doc/scipy/reference/linalg.html) which builds on NumPy. Let's import that packages:

import scipy.linalg as la

### Inverse

We can find the inverse using the function `scipy.linalg.inv`:

A = np.array([[1,2],[3,4]])
print(A)

la.inv(A)

### Determinant

We find the determinant using the function `scipy.linalg.det`:

A = np.array([[1,2],[3,4]])
print(A)

la.det(A)

### Example

#### Characteristic Polynomials and Cayley-Hamilton Theorem

The characteristic polynomial of a 2 by 2 square matrix $A$ is

$$
p_A(\lambda) = \det(A - \lambda I) = \lambda^2 - \mathrm{tr}(A) \lambda + \mathrm{det}(A)
$$

The [Cayley-Hamilton Theorem](https://en.wikipedia.org/wiki/Cayley%E2%80%93Hamilton_theorem) states that any square matrix satisfies its characteristic polynomial. For a matrix $A$ of size 2, this means that

$$
p_A(A) = A^2 - \mathrm{tr}(A) A + \mathrm{det}(A) I = 0
$$

Let's verify the Cayley-Hamilton Theorem for a few different matrices.

print(A)

trace_A = np.trace(A)
det_A = la.det(A)
I = np.eye(2)
A @ A - trace_A * A + det_A * I

Let's do this again for some random matrices:

N = np.random.randint(0,10,[2,2])
print(N)

trace_N = np.trace(N)
det_N = la.det(N)
I = np.eye(2)
N @ N - trace_N * N + det_N * I

Next time we will see how to solve a matrix equation $Ax = b$ in NumPy and SciPy

### Exercise

Compute the matrix equation $AB + 2B^2 - I$ for matrices $A = \begin{bmatrix} 3 & 4 \\\ -1 & 2 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 2 \\\ 8 & -3 \end{bmatrix}$.              