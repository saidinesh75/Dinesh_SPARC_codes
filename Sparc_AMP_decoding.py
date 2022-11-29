import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!! (info and warnings are not printed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np 
import matplotlib.pyplot as plt

import math
import sys
import numpy.linalg as la
import numpy.matlib
import tensorflow.compat.v1 as tf

rng = np.random.RandomState(seed=None)
import pickle

from eta import eta
from power_dist import power_dist
from generate_message import generate_message 
from tau_calculate import tau_calculate

# System parameters

system_params   = {'P': 10.0,   # Average codeword symbol power constraint
                 'R': 0.5,      # Rate
                 'L': 32,      # Number of sections
                 'M': 512,       # Columns per section
                 #'awgn_var':1,  # variance of noise
                 'snr_rx': 31,  # SNR at the receiver  (not in dB)   (SNR of 15 gives a capacity of 2 bits/sec)
                 'dist': 0      # 0 for flat and 1 for exponential
                }


system_params_list = ['P','R','L','M','snr_rx']
P,R,L,M,snr_rx = map(system_params.get, system_params_list)
N = L*M 

# Generating the message vectors
cols = 100
beta, beta2, beta_val, beta2_val, new_system_params, SNR_params = generate_message(system_params,rng,cols)

# Extracting the new system parameters
new_system_params_list = ['n','R_actual','delim','P_vec']
n, R, delim,P_vec = map(new_system_params.get, new_system_params_list)


# Extracting the SNR params
SNR_params_list = ['snr_rx','awgn_var','capacity']
# SNR_params_list = ['snr_rx','capacity']
snr_rx,awgn_var,capacity = map(SNR_params.get, SNR_params_list)
sigma = np.sqrt(awgn_var)


# Matrix generation
A_unn = np.random.normal(size=(n, N), scale=1.0 / math.sqrt(n)).astype(np.float32)
A = A_unn/np.linalg.norm(A_unn,axis=0) 
y = np.matmul(A,beta2) + np.random.normal( size = (n,cols), scale = math.sqrt(awgn_var) )


# AMP Decoding
T_star = int(np.ceil((2*capacity)/np.log2(capacity/R)) + 1)
tau_vec= tau_calculate(awgn_var, delim,P,P_vec,T_star,n)

beta_T = np.zeros((N,cols))
beta_hat = np.zeros((N,cols))
z = np.zeros((n,cols))

for t in range(T_star):
    z = y - np.matmul(A, beta_hat) + (z/tau_vec[t])*(P - (np.linalg.norm(beta_hat)**2)/n )
    beta_hat = beta_hat + np.matmul(A.T,z)
    beta_hat = eta(beta_hat,P_vec, tau_vec[t],delim)
    
for j in range(L):
    loc = np.argmax( beta_hat[ int(delim[0,j]) : int(delim[1,j]) ], axis=0 )
    for k in range(np.size(loc)):
        beta_T[ int(delim[0,j]) + loc[k],k ] = 1

check = beta-beta_T
check_sum = np.sum(np.absolute(check),axis=0)/2
check_avg = int(np.average(check_sum))
output_params = system_params.copy()
output_params['check_avg']=check_avg
