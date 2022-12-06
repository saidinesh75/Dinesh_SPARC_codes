import numpy as np
from power_dist import power_dist 

def generate_message(system_params,rng,cols):
    system_params_list = ['P','R','L','M','snr_rx','dist']
    P,R,L,M,snr_rx,dist = map(system_params.get, system_params_list) 
    N = L*M

    # Sparse Message vectors with ones (number of msg vectors=cols)
    beta = np.zeros((L*M,cols))
    beta_val = np.zeros((L*M, cols))

    # Same Sparse Message vectors with power allocated non-zero values
    beta2 = np.zeros((L*M,cols))
    beta2_val = np.zeros((L*M, cols))

    #Section sizes 
    bit_len = int(round(L*np.log2(M)))
    logM = int(round(np.log2(M)))
    sec_size = logM
    L = bit_len // sec_size

    # Actual rate
    n = int(round(bit_len/R))
    R_actual = bit_len / n      

    # SNR and noise variance calculation
    #snr_rx = np.power(10., snr_rx/10.) * (N/n)              # *log2M (not sure if it should be there)  # according to LISTA code
    #snr_rx = P/awgn_var

    capacity = 0.5 * np.log2(1 + snr_rx)
    
    awgn_var = P/snr_rx
    sigma = np.sqrt(awgn_var)
    
    SNR_params = {'snr_rx':snr_rx,
                 'awgn_var':awgn_var,
                 'capacity':capacity}

    # Power allocation
    power_dist_params = {'P':P,
                        'L':L,
                        'n':n,
                        'capacity':capacity,
                        'dist':dist,        # dist=0 for flat and 1 for exponential and 2 for modified PA
                        'R': R_actual
                         }
    beta_non_zero_values, P_vec = power_dist(power_dist_params)

    for i in range(2*cols):
        bits_in = rng.randint(2, size=bit_len)
        beta0 = np.zeros(L*M)   #placeholder for beta and beta_val (nonzeros =1)
        beta1 = np.zeros(L*M)   #placeholder for beta2 and beta2_val (nonzeros = power allocated values)

        for l in range(L):
            bits_sec = bits_in[l*sec_size : l*sec_size + logM]
            assert 0<logM<64
            idx = bits_sec.dot(1 << np.arange(logM)[::-1])
            beta0[l*M + idx] = 1
            beta1[l*M + idx] = beta_non_zero_values[l]

        # storing first 'cols' vectors in beta(beta2) and the next 'cols' vectors in beta_val(beta2_val)
        if i < cols:
            beta[:,i-1] = beta0
            beta2[:,i-1] = beta1
        else:
            beta_val[:,i - cols] = beta0
            beta2_val[:, i - cols] = beta1

        delim = np.zeros([2,L])
        delim[0,0] = 0
        delim[1,0] = M-1

        for i in range(1,L):
            delim[0,i] = delim[1,i-1]+1
            delim[1,i] = delim[1,i-1]+M

    new_system_params = {'n':n,
                        'R_actual': R_actual,
                        'delim': delim,
                        'P_vec':P_vec
                        }
                        
    return beta, beta2, beta_val, beta2_val, new_system_params,SNR_params, 
