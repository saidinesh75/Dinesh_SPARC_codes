import numpy as np

def power_dist(power_dist_params):

    power_dist_params_list = ['P','L','n','capacity','dist','R']
    P,L,n,capacity,dist, R =  map(power_dist_params.get, power_dist_params_list) 

    ##### Flat power Distribution
    P_vec_flat = P/L * np.ones((L,1))  # Considering Flat power distribution. Can also do exponential distribution


    ##### Exponential Power distribution
    #kappa = 2*capacity
    P_vec_unnorm = np.zeros(L)
    for i in range(L):
        P_vec_unnorm[i] = np.power(2, (-2*R*i)/L )
    pow_exp = np.sum(P_vec_unnorm)
    P_vec_exp = (P/pow_exp) * P_vec_unnorm    # Normalising such that the sum of Pi's = P


    ##### Modified Power allocation
    P_vec_unnorm_mod = np.zeros(L)
    for i in range(L):
        P_vec_unnorm_mod[i] = P * ( (np.power(2, (2*capacity)/L ) - 1)/(1 - (np.power(2,-2*capacity)))) * np.power(2, (-2*capacity*i)/L )
    pow_exp_mod = np.sum(P_vec_unnorm)
    P_vec_mod = (P/pow_exp_mod) * P_vec_unnorm_mod
    
    if dist==0:
        beta_non_zero_values = np.sqrt(n*P_vec_flat)
        P_vec = P_vec_flat
    elif dist==1:
        beta_non_zero_values = np.sqrt(n*P_vec_exp).reshape((L,1))      #use for exponential power distribution
        P_vec = P_vec_exp.reshape((L,1))
    else:
        beta_non_zero_values = np.sqrt(n*P_vec_mod).reshape((L,1))      #use for exponential power distribution
        P_vec = P_vec_mod.reshape((L,1))

    return beta_non_zero_values,P_vec