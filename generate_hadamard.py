import numpy as np
from scipy.linalg import hadamard

def generate_hadamard(N,n,rng):
    # m = np.log2(N)
    A_h = hadamard(N) 
    [rows,columns] = np.shape(A_h)
    order = np.arange(rows)
    rng.shuffle(order)
    final_rows = order[:n]
    A_hadamard = A_h[final_rows,:]

    return A_hadamard


