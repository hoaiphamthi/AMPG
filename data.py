import numpy as np

'''
This file generates data for the constraints of function H.
Hj(x) := max_z  {x.T * z},   j =1,...,m
Zj = {z \\in R^2n: Ajz <= bj}
Aj \\in R^dxn, bj \\in R^dnếu b


This function returns:
- dimA: array of size (m) containing all values 2*n ??? WHY
- A_list: list containing m matrices A_j (size 2n x n)
- b_list: list containing m vectors b_j (size 2n x 1)
'''

def data(n, m, delta):

    np.random.seed(42)

    # dimA(1:m) = 2*n => each wi has 2n rows => total number of w variables is 2n * m: since wi is a dual variable Ai.T * wi = p     
    dimA = np.full(m, 2 * n) 
    
    # lb and ub for matrix A ?? why use -10 and 10 ???
    l = -10  
    u = 10
    
    # Initialize list (replaces MATLAB cell array)
    A_list = []
    b_list = []
    
    # for ind = 1:m (Loop m times)
    for _ in range(m):
        # Generate random matrix A(n, n)
        # Since the set Z chosen by Bello has the form Z = {\delta * e <= Bjz <= delta * e} = {Aj * z <= bj} (A size m * n)
        # =>> A has size 2n * n   

        A = l + np.random.rand(n, n) * (u - l)
        # A_matrix = [A; -A]; (Concatenate A and -A vertically)
        
        # create symmetric positive semi-definite matrix A (bello writes A is invertible)
        A_matrix = np.vstack((A, -A))
        A_list.append(A_matrix)
        
        # b = delta * np.ones((2n, 1));
        b_vector = delta * np.ones((2 * n, 1))
        b_list.append(b_vector)
        
    return dimA, A_list, b_list

