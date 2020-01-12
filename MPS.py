import numpy as np
from numpy import tensordot
# set the bond dimensions of the varius legs
Da, Db, Dc, Dd, De, Df = 2,3,3,2,3,5
# Initialize the A,B tensors using random values
A = np.random.normal(size=[Da, Db, Dc, Dd])
B = np.random.normal(size=[Dc, Dd, De, Df])
# contract A,B along the c,d legs. Resulting tensor will have the
# legs a,b,e,f (in the order)
C = tensordot(A, B, axes=([2,3], [0,1]))
# permute the order of the legs of C so that they will be in the order
# a,e,b,f
C1 = C.transpose([0,2,1,3])
# Fuse the a,e legs and the b,f legs into two big legs
Cprime = C1.reshape([Da*De, Db*Df])
# print the first entries of the matrix Cprime:
print(Cprime[0:2,0:2])


def concate_tensors(tensor_a, tensor_b, leg_to_sum_a, leg_to_sum_b):
    return tensordot(tensor_a, tensor_b, axes=([leg_to_sum_a], [leg_to_sum_b]))


def sum_left_side_with_next_A_old(left_side, A, with_B=False):
    # start computing from left to right (all the tensors hold real values so conjugate is unnessecery
    temp = concate_tensors(left_side, A, 0, 1)
    print(temp.size)
    temp = concate_tensors(temp, A, 0, 1)
    print(temp.size)
    if with_B:
        temp = concate_tensors(B, temp, 0, 0)
    print(concate_tensors(temp, A, 0, 2).size)
    return concate_tensors(temp, A, 0, 2)

def sum_left_side_with_next_A(left_side, A):
    # start computing from left to right (all the tensors hold real values so conjugate is unnessecery
    new_A = tensordot(A, A, axes=([0], [0]))
    return tensordot(left_side, new_A, axes=([0, 1], [0, 2]))

def sum_left_side_with_B_and_next_A(left_side, A, B):
    # start computing from left to right (all the tensors hold real values so conjugate is unnessecery
    temp_a = tensordot(B, A, axes=([0], [0]))
    new_A = tensordot(temp_a, A, axes=([0], [0]))
    return tensordot(left_side, new_A, axes=([0, 1], [0, 2]))

N = 10

D = 2
# Initialize the A,B tensors using random values
A = np.random.normal(size=[D, D, D])
L = np.random.normal(size=[D, D])
R = np.random.normal(size=[D, D])

B = [[1, 0.5], [0.5, -2]]
#B = [[1, 0], [0, 1]]
# build the tensors
for j in range(2):
    for a in range(2):
        for b in range(2):
            A[j][a][b] = np.sin(2*j + a) - 0.7*b
for j in range(2):
    for a in range(2):
        L[j][a] = np.cos(j*a)
        # for R: a==b
        R[j][a] = j + a

answer = []
for N in range(400, 410, 10):
    print(N)
    # temp: hold a Tensor which keeps the computation
    temp = concate_tensors(L, L, 0, 0)
    n = (2*N)-1
    # first, calculate the <psi|B|psi>
    for i in range(1, n-1):
        if i == N:
            temp = sum_left_side_with_B_and_next_A(temp, A, B)
        else:
            temp = sum_left_side_with_next_A(temp, A)
    # sum with the Rs
    new_R = concate_tensors(R, R, 0, 0)
    result = tensordot(temp, new_R, axes=([0, 1], [0, 1]))

    # calculate norm
    temp = concate_tensors(L, L, 0, 0)
    for i in range(1, n-1):
        temp = sum_left_side_with_next_A(temp, A)
    # sum with the Rs
    new_R = concate_tensors(R, R, 0, 0)
    temp_num = tensordot(temp, new_R, axes=([0, 1], [0, 1]))
    norm = np.abs(temp_num) #** 2

    print(result)
    answer.append(result/norm)
print(answer)
