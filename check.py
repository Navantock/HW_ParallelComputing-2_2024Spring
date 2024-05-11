# Foundations of Parallel Computing II, Spring 2024.
# Instructor: Chao Yang, Xiuhong Li @ Peking University.
import numpy as np
import sys
eps = 1e-6

def read_matrix(file_name):
    with open(file_name, 'r') as f:
        n = int(f.readline())
        matrix = np.zeros((n, n))
        for i in range(n):
            matrix[i] = list(map(float, f.readline().split()))
    return matrix, n

def read_LU(file_name, n):
    with open(file_name, 'r') as f:
        f.readline()  # skip the line "L matrix:"
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        for i in range(n):
            L[i] = list(map(float, f.readline().split()))
        f.readline()  # skip the line "U matrix:"
        for i in range(n):
            U[i] = list(map(float, f.readline().split()))
    return L, U

def is_lower_triangular(matrix):
    return np.allclose(matrix, np.tril(matrix), rtol=eps)

def is_upper_triangular(matrix):
    return np.allclose(matrix, np.triu(matrix), rtol=eps)

def is_non_singular(matrix):
    return np.linalg.det(matrix) != 0

if len(sys.argv) != 3:
    print("Usage: python <script_name> <input_matrix_file> <LU_file>")
    sys.exit(1)

input_matrix_file = sys.argv[1]
LU_file = sys.argv[2]

A, n = read_matrix(input_matrix_file)
L, U = read_LU(LU_file, n)

print("L is non-singular:", is_non_singular(L))
print("U is non-singular:", is_non_singular(U))
print("L is lower triangular:", is_lower_triangular(L))
print("U is upper triangular:", is_upper_triangular(U))

LU = np.dot(L, U)
abs_diff = np.abs(A - LU)
max_diff = np.max(abs_diff)
percent_90_diff = np.percentile(abs_diff, 90)
print("The maximum value of the difference:", max_diff)
print("The 90th percentile of the difference:", percent_90_diff)
