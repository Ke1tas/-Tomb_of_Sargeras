import numpy as np
import pandas as pd
from scipy.linalg import lu, solve

# №1
A = np.array([
    [2, 3, 1, 4, 5],
    [0, 3, 2, 1, 4],
    [0, 0, 4, 3, 2],
    [0, 0, 0, 2, 1],
    [0, 0, 0, 0, 3]
], dtype=float)

B = np.array([15, 10, 12, 8, 9], dtype=float)

x = np.linalg.solve(A, B)
print('Matrix A: ', A)
print('Vector: ', B)
print('Solution: ', x)
print('Correctness: B = ', A @ x)

# №2
A2 = np.array([
    [4.3, -12.1, 23.2, -14.1],
    [2.4, -4.4, 3.5, 5.5],
    [5.4, 8.3, -7.4, -12.7],
    [6.3, -7.6, 1.34, 3.7]
])

B2 = np.array([15.5, 2.5, 8.6, 12.1])

solving = lu(A2)
print('P: ', solving[0])
print('L: ', solving[1])
print('U: ', solving[2])

# PLUX = B LUX = Z UX = Y
Z = solve(solving[0], B2)
Y = solve(solving[1], Z)
# UX = Y
X2 = solve(solving[2], Y)
print('solution: ', X2)
print('Correctness: B = ', A2 @ X2)
print('Correctness: A = ', solving[0] @ solving[1] @ solving[2])

# №3
Q, R = np.linalg.qr(A2)
print('Q: ', Q)
print('R: ', R)
Y3 = np.linalg.solve(Q, B2)
X3 = np.linalg.solve(R, Y3)
print(A2 @ X3)


# №4
# x^(n) = Bx^(n-1) + c

def to_upper_triangular_fast(A, pivoting=True):
    """
    Быстрая версия для практического использования
    """
    n = A.shape[0]
    A = A.astype(float).copy()

    if pivoting:
        for k in range(n - 1):
            # Выбор главного элемента
            pivot = k + np.argmax(np.abs(A[k:, k]))
            if pivot != k:
                A[[k, pivot]] = A[[pivot, k]]

            # Исключение
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
    else:
        for k in range(n - 1):
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]

    # Обнуляем нижнюю треугольную часть для чистоты
    for i in range(1, n):
        for j in range(i):
            A[i, j] = 0

    return A


A4 = np.array([
    [3.2, -2.5, 3.7],
    [0.5, 0.34, 1.7],
    [1.6, 2.3, -1.5]
])
A4_tryang = to_upper_triangular_fast(A4)
print(A4_tryang)
A4 = A4_tryang
A4[0,1] = 0
A4[0,2] = 0
print('\n',A4,'\n')
B4 = np.array([6.5, -0.24, 4.3])
n = A4.shape[0]
C = np.zeros((n, n))
d = np.zeros(n)
for i in range(n):
    d[i] = B4[i] / A4[i, i]
    for j in range(n):
        if i != j:
            C[i, j] = -A4[i, j] / A4[i, i]
print("Matrix C :", C)
print("Vector d: ", d)
norm_C = np.linalg.norm(C)
print(norm_C)
while norm_C > 1:
    print('||C|| > 1: ', norm_C, '> 1')
    '''temp = A4[0].copy()
    for i in range(n-1):
        A4[i] = A4[i+1]
    A4[n-1] = temp
    print(A4)'''
    '''temp = A4[0].copy()
    for i in range(n-1):
        A4[i] -= A4[i+1]
    A4[n - 1] -= temp'''
    # A4 = np.fliplr(A4)
    '''A4 = A4[:,[0,2,1]]
    A4 = A4[:, [2, 0, 1]]'''
    print(A4)
    for i in range(n):
        d[i] = B4[i] / A4[i, i]
        for j in range(n):
            if i != j:
                C[i, j] = -A4[i, j] / A4[i, i]
    print("Matrix C :", C)
    print("Vector d: ", d)
    norm_C = np.linalg.norm(C)
    print('\nnew norm\n', norm_C)
n = len(d)
x = np.zeros(n)
error = 1
print()
while error > 1e-3:
    x_new = C @ x + d
    error = np.max(np.abs(x_new - x))
    print(error)
    x = x_new.copy()
print('solution: ', x)
print('Correctness: B = ', A4 @ x)

# 5

A5 = np.array([
    [3.2, -2.5, 3.7],
    [0.5, 0.34, 1.7],
    [1.6, 2.3, -1.5],
    [3.6, 1.8, -4.7]
])
B5 = np.array([6.5, -0.24, 4.3, 3.8])

print("Матрица A (4x3):")
print(A5)
print("\nВектор B:", B5)

A5_pseudo = A5.T @ A5
B5_pseudo = A5.T @ B5
A5_pseudo_inv = np.linalg.inv(A5_pseudo)

X5 = solve(np.eye(3, 3), A5_pseudo_inv @ B5_pseudo)
print(f"\nПсевдорешение: {X5}")

# Проверка невязки
residual = A5 @ X5 - B5
print(f"Невязка: {residual}")
print(f"Норма невязки: {np.linalg.norm(residual):.6f}")
