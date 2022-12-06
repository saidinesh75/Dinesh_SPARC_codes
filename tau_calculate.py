import numpy as np

def tau_calculate(sigma_sq,delim, P, P_vec, T_star, n,rng):
    tau_vec = np.zeros([T_star,1])
    x = np.zeros([T_star,1])
    tau_vec[0] = sigma_sq + P
    ITR = 10^4

    for t in range(1,T_star):
        exp_vec = np.zeros([np.size(P_vec),1])

        for i in range(ITR):
            U = rng.randn(int(delim[-1,-1]))
            for l in range(np.size(P_vec)):
                # Numerator
                num_1 =  np.sqrt(n*P_vec[l],dtype=np.float128)/tau_vec[t-1]
                num_21 = U[int(delim[0,l])]
                num = np.exp( num_1 * (num_1 + num_21 ),dtype=np.float128)

                # Denominator
                denom_21 = U[ int(delim[0,l]+1) : int(delim[1,l]+1) ]
                denom_2 = np.sum( np.exp( np.multiply(num_1,denom_21) ), dtype=np.float128)
                denom = num + denom_2

                exp_vec[l] = exp_vec[l] + np.divide(num,denom, dtype=np.float128)

        exp_vec = np.divide(exp_vec,ITR)
        x1 = np.ndarray.item(np.float128(np.matmul(P_vec.T,exp_vec)/P)) # getting saved as an (1,1) ndarray
        
        tau_vec[t] = sigma_sq + P*(1-x1)
        x[t] = x1

    return tau_vec, x        