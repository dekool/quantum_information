import numpy as np
from numpy import tensordot

def sum_left_side_with_next_A(left_side, A):
    """
    join both A matrices and than contract them with the left side of the state
    :return: new tensor representing the left side of the computation
    """
    # all the tensors hold real values so conjugate is unnecessary
    new_A = tensordot(A, A, axes=([0], [0]))
    return tensordot(left_side, new_A, axes=([0, 1], [0, 2]))

def sum_left_side_with_B_and_next_A(left_side, A, B):
    """
    first join the upper A with the B matrix, and than contract the result with the bottom A
    and than contract the result with the left side of the state
    :return: new tensor representing the left side of the computation
    """
    # all the tensors hold real values so conjugate is unnecessary
    temp_a = tensordot(B, A, axes=([0], [0]))
    new_A = tensordot(temp_a, A, axes=([0], [0]))
    return tensordot(left_side, new_A, axes=([0, 1], [0, 2]))

def initialize_matrices():
    D = 2
    # Initialize the A,L,R tensors using random values just for fixing their sizes easily
    A = np.random.normal(size=[D, D, D])
    L = np.random.normal(size=[D, D])
    R = np.random.normal(size=[D, D])

    B = [[1, 0.5], [0.5, -2]]
    # build the tensors
    for j in range(2):
        for a in range(2):
            for b in range(2):
                A[j][a][b] = np.sin(2 * j + a) - 0.7 * b
    for j in range(2):
        for a in range(2):
            L[j][a] = np.cos(j * a)
            # for R: a==b
            R[j][a] = j + a
    return A, B, L, R

def calculate_psi_B_psi(N):
    """
    calculates <psi|B|psi> from left side to the right. holds maximum 6 "open legs" at a given time
    :return: number - the result of <psi|B|psi>
    """
    n = (2 * N) - 1
    # temp: hold a Tensor which keeps the computation
    temp = tensordot(L, L, axes=([0], [0]))

    for i in range(1, n - 1):
        if i == N:
            temp = sum_left_side_with_B_and_next_A(temp, A, B)
        else:
            temp = sum_left_side_with_next_A(temp, A)
    # sum with the Rs
    new_R = tensordot(R, R, axes=([0], [0]))
    return tensordot(temp, new_R, axes=([0, 1], [0, 1]))

def calculate_norm(N):
    """
    calculate the norm of the state. doing than using tensors - from left to right.
    holds maximum 6 "open legs" at a given time
    :return: the norm of psi
    """
    n = (2 * N) - 1
    # temp: hold a Tensor which keeps the computation
    temp = tensordot(L, L, axes=([0], [0]))

    for i in range(1, n - 1):
        temp = sum_left_side_with_next_A(temp, A)
    # sum with the Rs
    new_R = tensordot(R, R, axes=([0], [0]))
    temp_num = tensordot(temp, new_R, axes=([0, 1], [0, 1]))
    return np.abs(temp_num)

A,B,L,R = initialize_matrices()
answer = {}
for N in range(10, 110, 10):
    result = calculate_psi_B_psi(N)
    norm = calculate_norm(N)

    answer[N] = result/norm
print(answer)
