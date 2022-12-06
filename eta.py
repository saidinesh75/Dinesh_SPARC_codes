import numpy as np 

def eta(beta_hat, P_vec, tau, delim,n):
    beta_th = np.zeros(np.shape(beta_hat))
    rows,cols = beta_hat.shape 
    L = np.size(P_vec)
    M = rows/L

    for i in range(cols):
        s1 = beta_hat[:,i]
        for j in range(L):
            beta_section = s1[ int(delim[0,j]):int(delim[1,j]+1)]
            beta_th_section = np.zeros(int(M))
            denom = np.sum( np.exp( (beta_section* np.sqrt(n*P_vec[j]))/(tau**2) ) )
            for k in range(int(M)):
                num = np.exp( (beta_section[k]* np.sqrt(n*P_vec[j]))/(tau**2) )
                beta_th_section[k] = num/denom
            beta_th[int(delim[0,j]):int(delim[1,j]+1), i] = beta_th_section

    return beta_th        